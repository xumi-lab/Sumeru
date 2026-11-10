//! Procedural macros for Sumeru 16KB page-aligned GPU compute (`#[gpu]`, `#[gpu_component]`, `#[gpu_kernel]`).
//!
//! Lowers Rust `syn` AST into Sumeru's target-independent Typed Shader IR (`GpuBlock` / `GpuStmt` / `GpuExpr`)
//! with lexical scope tracking, compile-time `const` evaluation (`GpuConstValue`), runtime dynamic scalar
//! uniforms (`GpuScalarUniformSpec`), and composable `#[gpu]` helper functions (`GpuHelperFnDescriptor`).

use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    BinOp, Block, Data, DeriveInput, Error, Expr, Fields, FnArg, GenericParam, Ident, Item,
    ItemFn, ItemStruct, Lit, Member, Pat, RangeLimits, ReturnType, Stmt, Type, UnOp,
};

/// Unified GPU attribute macro for Sumeru ECS.
///
/// - When placed on a `struct`, expands to a `#[repr(C)]` Pod/Zeroable GPU component with
///   automatic `GpuComponent` and `GpuShaderStruct` (MSL + WGSL) implementations.
/// - When placed on a `fn` with `1..=8` `&T` / `&mut T` component parameters (and optional trailing
///   runtime scalar uniform parameters), lowers the Rust AST into a Typed Shader IR (`GpuBlock`)
///   and expands to a three-tier operator (`run`, `par_run`, `gpu().for_each`).
/// - When placed on a helper `fn` (returning a scalar or taking array/scalar references), expands to
///   a composable `#[gpu]` device helper function callable on both CPU and GPU.
#[proc_macro_attribute]
pub fn gpu(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_gpu_item(attr.into(), item.into()).into()
}

/// Unified heterogeneous compute & component attribute macro for Sumeru ECS (`#[kernel]`).
///
/// - On a `struct`: defines a 16KB page-aligned `#[repr(C)]` POD component with automatic
///   `GpuComponent` and `GpuShaderStruct` (MSL + WGSL) implementations.
/// - On a `fn`: defines a unified operator supporting row-level calls (`integrate(p, v, dt)`
///   inside Rayon loops or composable `#[kernel]` functions) and whole-column execution
///   (`integrate.run(...)`, `integrate.par(...)`, `integrate.gpu(...)`, `integrate.bind(...)`).
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_gpu_item(attr.into(), item.into()).into()
}

/// Attribute macro for GPU-compatible `#[repr(C)]` ECS components.
#[proc_macro_attribute]
pub fn gpu_component(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_gpu_component_impl(attr.into(), item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Attribute macro for multi-platform GPU compute kernels (MSL + WGSL + CPU).
#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: TokenStream2 = attr.into();
    let has_custom_type_name = !attr2.is_empty();
    expand_gpu_fn_impl(attr2, item.into(), !has_custom_type_name)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive macro for `GpuComponent`.
#[proc_macro_derive(GpuComponent)]
pub fn derive_gpu_component(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as DeriveInput);
    match derive_gpu_component_impl(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Derive macro for `GpuShaderStruct` (generates MSL & WGSL struct definitions).
#[proc_macro_derive(GpuShaderStruct)]
pub fn derive_gpu_shader_struct(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as DeriveInput);
    match derive_gpu_shader_struct_impl(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_gpu_item(attr: TokenStream2, item: TokenStream2) -> TokenStream2 {
    match syn::parse2::<Item>(item.clone()) {
        Ok(Item::Struct(_)) => {
            expand_gpu_component_impl(attr, item).unwrap_or_else(|e| e.to_compile_error())
        }
        Ok(Item::Fn(_)) => {
            expand_gpu_fn_impl(attr, item, true).unwrap_or_else(|e| e.to_compile_error())
        }
        Ok(other) => Error::new_spanned(
            other,
            "`#[gpu]` can only be applied to a `struct` or `fn` item",
        )
        .to_compile_error(),
        Err(e) => e.to_compile_error(),
    }
}

fn derive_gpu_component_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &input.generics,
            "`GpuComponent` cannot have generic parameters",
        ));
    }
    let Data::Struct(_) = &input.data else {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuComponent)]` can only be applied to a struct",
        ));
    };
    let name = &input.ident;
    Ok(quote! {
        unsafe impl ::sumeru::GpuComponent for #name {}
    })
}

fn derive_gpu_shader_struct_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &input.generics,
            "`GpuShaderStruct` cannot have generic parameters",
        ));
    }
    let Data::Struct(data_struct) = &input.data else {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuShaderStruct)]` can only be applied to a named struct",
        ));
    };
    let Fields::Named(fields_named) = &data_struct.fields else {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuShaderStruct)]` requires named struct fields `{ ... }`",
        ));
    };
    if fields_named.named.is_empty() {
        return Err(Error::new_spanned(
            input,
            "`#[derive(GpuShaderStruct)]` struct must have at least one field",
        ));
    }
    let name = &input.ident;
    let name_str = name.to_string();
    let field_entries = fields_named.named.iter().map(|f| {
        let fname = f.ident.as_ref().unwrap().to_string();
        let fty = &f.ty;
        quote! {
            ::sumeru::GpuStructField {
                name: #fname,
                field_type: <#fty as ::sumeru::GpuScalarType>::FIELD_TYPE,
            }
        }
    });
    Ok(quote! {
        impl ::sumeru::GpuShaderStruct for #name {
            const STRUCT_NAME: &'static str = #name_str;
            const FIELDS: &'static [::sumeru::GpuStructField] = &[
                #(#field_entries),*
            ];
        }
    })
}

fn expand_gpu_component_impl(attr: TokenStream2, item: TokenStream2) -> syn::Result<TokenStream2> {
    if !attr.is_empty() {
        return Err(Error::new_spanned(
            attr,
            "`#[gpu]` on a struct does not accept attribute arguments",
        ));
    }
    let item_struct: ItemStruct = syn::parse2(item)?;
    if !item_struct.generics.params.is_empty() || item_struct.generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &item_struct.generics,
            "`#[gpu]` component structs cannot have generic parameters or where clauses",
        ));
    }
    let Fields::Named(fields_named) = &item_struct.fields else {
        return Err(Error::new_spanned(
            &item_struct,
            "`#[gpu]` on a type requires a named struct with `{ ... }` fields",
        ));
    };
    if fields_named.named.is_empty() {
        return Err(Error::new_spanned(
            &item_struct,
            "`#[gpu]` component structs cannot be zero-sized (must have at least one field)",
        ));
    }

    let attrs: Vec<_> = item_struct
        .attrs
        .iter()
        .filter(|a| !a.path().is_ident("repr"))
        .collect();
    let vis = &item_struct.vis;
    let name = &item_struct.ident;
    let fields = fields_named.named.iter().map(|f| {
        let fattrs = &f.attrs;
        let fvis = &f.vis;
        let fident = f.ident.as_ref().unwrap();
        let fty = &f.ty;
        quote! {
            #(#fattrs)*
            #fvis #fident : #fty
        }
    });

    Ok(quote! {
        ::sumeru::sumeru_gpu_struct! {
            #(#attrs)*
            #vis struct #name {
                #(#fields),*
            }
        }
    })
}

struct ParsedKernelParam {
    ident: Ident,
    is_mut: bool,
    elem_ty: Type,
}

struct ParsedUniformParam {
    ident: Ident,
    is_mut: bool,
    ty: Type,
    kind_tokens: TokenStream2,
}

struct ParsedHelperParam {
    ident: Ident,
    is_mut_binding: bool,
    ty: Type,
    spec_tokens: TokenStream2,
}

fn is_unit_type(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(t) if t.elems.is_empty())
}

fn is_scalar_type_name(s: &str) -> bool {
    matches!(s.trim(), "f32" | "i32" | "u32" | "usize" | "bool")
}

fn parse_array_type_info(ty: &Type) -> Option<(TokenStream2, usize)> {
    match ty {
        Type::Array(arr) => {
            let elem_kind = parse_scalar_kind_tokens(&arr.elem).ok()?;
            let Expr::Lit(expr_lit) = &arr.len else {
                return None;
            };
            let Lit::Int(il) = &expr_lit.lit else {
                return None;
            };
            let len: usize = il.base10_parse().ok()?;
            if (2..=4).contains(&len) {
                Some((elem_kind, len))
            } else {
                None
            }
        }
        Type::Path(tp) if tp.qself.is_none() => {
            let last = tp.path.segments.last()?.ident.to_string();
            match last.as_str() {
                "Vec2" => Some((quote! { ::sumeru::GpuScalarKind::F32 }, 2)),
                "Vec3" => Some((quote! { ::sumeru::GpuScalarKind::F32 }, 3)),
                "Vec4" => Some((quote! { ::sumeru::GpuScalarKind::F32 }, 4)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_component_struct_ref(ty: &Type) -> bool {
    let Type::Reference(ty_ref) = ty else {
        return false;
    };
    match &*ty_ref.elem {
        Type::Reference(_) | Type::Array(_) | Type::Slice(_) | Type::Tuple(_) => false,
        other => {
            !is_scalar_type_name(&other.to_token_stream().to_string())
                && parse_array_type_info(other).is_none()
        }
    }
}

fn validate_fn_common_signature(sig: &syn::Signature) -> syn::Result<()> {
    if let Some(async_kw) = &sig.asyncness {
        return Err(Error::new_spanned(
            async_kw,
            "`#[gpu]` functions cannot be `async`",
        ));
    }
    if let Some(const_kw) = &sig.constness {
        return Err(Error::new_spanned(
            const_kw,
            "`#[gpu]` functions cannot be `const`",
        ));
    }
    if let Some(unsafe_kw) = &sig.unsafety {
        return Err(Error::new_spanned(
            unsafe_kw,
            "`#[gpu]` functions cannot be `unsafe`",
        ));
    }
    if let Some(abi) = &sig.abi {
        return Err(Error::new_spanned(
            abi,
            "`#[gpu]` functions cannot have an `extern` ABI",
        ));
    }
    if let Some(variadic) = &sig.variadic {
        return Err(Error::new_spanned(
            variadic,
            "`#[gpu]` functions cannot be variadic",
        ));
    }
    for param in &sig.generics.params {
        match param {
            GenericParam::Lifetime(_) => {}
            GenericParam::Type(tp) => {
                return Err(Error::new_spanned(
                    tp,
                    "`#[gpu]` functions cannot have generic type parameters",
                ));
            }
            GenericParam::Const(cp) => {
                return Err(Error::new_spanned(
                    cp,
                    "`#[gpu]` functions cannot have const generic parameters",
                ));
            }
        }
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        return Err(Error::new_spanned(
            where_clause,
            "`#[gpu]` functions cannot have a `where` clause",
        ));
    }
    Ok(())
}

fn expand_gpu_fn_impl(
    attr: TokenStream2,
    item: TokenStream2,
    same_name_callable: bool,
) -> syn::Result<TokenStream2> {
    let item_fn: ItemFn = syn::parse2(item)?;
    let sig = &item_fn.sig;
    validate_fn_common_signature(sig)?;

    // 1. Non-unit return type -> always a composable helper function
    if let ReturnType::Type(_, ret_ty) = &sig.output {
        if !is_unit_type(ret_ty) {
            if !same_name_callable || !attr.is_empty() {
                return Err(Error::new_spanned(
                    ret_ty,
                    "`#[gpu_kernel]` entry points must not have a return type; use `#[gpu]` for composable helper functions",
                ));
            }
            return expand_gpu_helper_fn_impl(&item_fn, Some(ret_ty));
        }
    }

    // 2. Unit return type, but first parameter is not a component struct reference (e.g. `&mut [f32; 3]` or `&mut f32`)
    if same_name_callable && attr.is_empty() && !sig.inputs.is_empty() {
        let first_is_comp_ref = match sig.inputs.first().unwrap() {
            FnArg::Typed(pt) => is_component_struct_ref(&pt.ty),
            FnArg::Receiver(_) => false,
        };
        let has_array_or_scalar_ref = sig.inputs.iter().any(|arg| match arg {
            FnArg::Typed(pt) => match &*pt.ty {
                Type::Reference(tr) => {
                    parse_array_type_info(&tr.elem).is_some()
                        || is_scalar_type_name(&tr.elem.to_token_stream().to_string())
                }
                Type::Array(_) => true,
                _ => false,
            },
            FnArg::Receiver(_) => false,
        });
        if !first_is_comp_ref || has_array_or_scalar_ref {
            return expand_gpu_helper_fn_impl(&item_fn, None);
        }
    }

    let mut comp_params = Vec::new();
    let mut uniform_params = Vec::new();
    let mut seen_scalar_uniform = false;

    for arg in &sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(Error::new_spanned(
                arg,
                "`#[gpu]` kernel functions cannot take `self`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(Error::new_spanned(
                &pat_type.pat,
                "`#[gpu]` parameter pattern must be a simple identifier",
            ));
        };
        let pname = pat_ident.ident.to_string();
        if pname.starts_with("__") || pname.starts_with("sumeru_") {
            return Err(Error::new_spanned(
                &pat_ident.ident,
                "`#[gpu]` parameter name cannot start with reserved prefix `__` or `sumeru_`",
            ));
        }
        match &*pat_type.ty {
            Type::Reference(ty_ref) => {
                if seen_scalar_uniform {
                    return Err(Error::new_spanned(
                        &pat_type.ty,
                        format!(
                            "component reference parameter `{pname}` must appear before trailing scalar uniform parameters in `#[gpu]` kernel"
                        ),
                    ));
                }
                if matches!(&*ty_ref.elem, Type::Reference(_)) {
                    return Err(Error::new_spanned(
                        &ty_ref.elem,
                        format!("parameter `{pname}` cannot be a nested reference"),
                    ));
                }
                comp_params.push(ParsedKernelParam {
                    ident: pat_ident.ident.clone(),
                    is_mut: ty_ref.mutability.is_some(),
                    elem_ty: (*ty_ref.elem).clone(),
                });
            }
            other_ty => {
                seen_scalar_uniform = true;
                let kind_tokens = parse_scalar_kind_tokens(other_ty)?;
                uniform_params.push(ParsedUniformParam {
                    ident: pat_ident.ident.clone(),
                    is_mut: pat_ident.mutability.is_some(),
                    ty: other_ty.clone(),
                    kind_tokens,
                });
            }
        }
    }

    if comp_params.is_empty() || comp_params.len() > 8 {
        return Err(Error::new_spanned(
            &sig.inputs,
            "`#[gpu]` kernel functions must have between 1 and 8 component reference parameters (`&T` or `&mut T`)",
        ));
    }
    if uniform_params.len() > 4 {
        return Err(Error::new_spanned(
            &sig.inputs,
            "`#[gpu]` kernel functions support at most 4 trailing runtime scalar uniform parameters",
        ));
    }

    let mut root_names: Vec<String> = comp_params.iter().map(|p| p.ident.to_string()).collect();
    for u in &uniform_params {
        root_names.push(u.ident.to_string());
    }
    let mut ir_builder = IrAstBuilder::new(root_names, false);
    let ir_block = ir_builder.lower_block(&item_fn.block)?;

    let attrs = &item_fn.attrs;
    let vis = &item_fn.vis;
    let fn_name = &sig.ident;
    let stmts = &item_fn.block.stmts;

    let comp_tokens: Vec<TokenStream2> = comp_params
        .iter()
        .map(|p| {
            let ident = &p.ident;
            let mode = if p.is_mut {
                Ident::new("mut", ident.span())
            } else {
                Ident::new("ref", ident.span())
            };
            let ty = &p.elem_ty;
            quote! { #ident : [#mode] #ty }
        })
        .collect();

    if uniform_params.is_empty() {
        if same_name_callable {
            if !attr.is_empty() {
                return Err(Error::new_spanned(
                    attr,
                    "`#[gpu]` on a function does not accept attribute arguments; use `#[gpu_kernel(CustomName)]` if needed",
                ));
            }
            Ok(quote! {
                ::sumeru::sumeru_gpu_fn! {
                    @ir
                    #(#attrs)*
                    #vis fn #fn_name ( #(#comp_tokens),* )
                    ir = #ir_block,
                    body = { #(#stmts)* }
                }
            })
        } else {
            let custom_ident: Ident = syn::parse2(attr)?;
            Ok(quote! {
                ::sumeru::sumeru_gpu_kernel! {
                    @ir
                    #(#attrs)*
                    #vis kernel #custom_ident fn #fn_name ( #(#comp_tokens),* )
                    ir = #ir_block,
                    body = { #(#stmts)* }
                }
            })
        }
    } else {
        let num_uniforms_lit = Literal::usize_unsuffixed(uniform_params.len());
        let uniform_tokens: Vec<TokenStream2> = uniform_params
            .iter()
            .enumerate()
            .map(|(slot_idx, u)| {
                let slot_lit = Literal::usize_unsuffixed(slot_idx);
                let ident = &u.ident;
                let mode = if u.is_mut {
                    Ident::new("mut", ident.span())
                } else {
                    Ident::new("ref", ident.span())
                };
                let ty = &u.ty;
                let kind = &u.kind_tokens;
                quote! { (#slot_lit) #ident : [#mode] #ty => #kind }
            })
            .collect();

        if same_name_callable {
            if !attr.is_empty() {
                return Err(Error::new_spanned(
                    attr,
                    "`#[gpu]` on a function does not accept attribute arguments; use `#[gpu_kernel(CustomName)]` if needed",
                ));
            }
            Ok(quote! {
                ::sumeru::sumeru_gpu_fn! {
                    @ir_with_uniforms (#num_uniforms_lit)
                    #(#attrs)*
                    #vis fn #fn_name (
                        components = [ #(#comp_tokens),* ],
                        uniforms = [ #(#uniform_tokens),* ]
                    )
                    ir = #ir_block,
                    body = { #(#stmts)* }
                }
            })
        } else {
            let custom_ident: Ident = syn::parse2(attr)?;
            Ok(quote! {
                ::sumeru::sumeru_gpu_kernel! {
                    @ir_with_uniforms (#num_uniforms_lit)
                    #(#attrs)*
                    #vis kernel #custom_ident fn #fn_name (
                        components = [ #(#comp_tokens),* ],
                        uniforms = [ #(#uniform_tokens),* ]
                    )
                    ir = #ir_block,
                    body = { #(#stmts)* }
                }
            })
        }
    }
}

fn expand_gpu_helper_fn_impl(item_fn: &ItemFn, ret_ty: Option<&Type>) -> syn::Result<TokenStream2> {
    let sig = &item_fn.sig;
    let (ret_ty_tokens, ret_kind_opt_tokens, has_return_val) = match ret_ty {
        Some(ty) => {
            let k = parse_scalar_kind_tokens(ty)?;
            (
                quote! { #ty },
                quote! { ::std::option::Option::Some(#k) },
                true,
            )
        }
        None => (
            quote! { () },
            quote! { ::std::option::Option::None },
            false,
        ),
    };

    let mut params = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(Error::new_spanned(
                arg,
                "`#[gpu]` helper functions cannot take `self`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(Error::new_spanned(
                &pat_type.pat,
                "`#[gpu]` helper parameter pattern must be a simple identifier",
            ));
        };
        let pname = pat_ident.ident.to_string();
        if pname.starts_with("__") || pname.starts_with("sumeru_") {
            return Err(Error::new_spanned(
                &pat_ident.ident,
                "`#[gpu]` parameter name cannot start with reserved prefix `__` or `sumeru_`",
            ));
        }

        let is_mut_binding = pat_ident.mutability.is_some();
        let (is_mut_in_shader, by_mut_ref, kind_tokens) = match &*pat_type.ty {
            Type::Reference(ty_ref) => {
                let ref_is_mut = ty_ref.mutability.is_some();
                if let Some((elem_kind, len)) = parse_array_type_info(&ty_ref.elem) {
                    (
                        ref_is_mut,
                        ref_is_mut,
                        quote! {
                            ::sumeru::GpuHelperParamKind::Array {
                                elem: #elem_kind,
                                len: #len,
                            }
                        },
                    )
                } else if is_scalar_type_name(&ty_ref.elem.to_token_stream().to_string()) {
                    let sk = parse_scalar_kind_tokens(&ty_ref.elem)?;
                    (
                        ref_is_mut,
                        ref_is_mut,
                        quote! { ::sumeru::GpuHelperParamKind::Scalar(#sk) },
                    )
                } else {
                    let comp_ty = &*ty_ref.elem;
                    (
                        ref_is_mut,
                        ref_is_mut,
                        quote! {
                            ::sumeru::GpuHelperParamKind::Struct {
                                struct_name: <#comp_ty as ::sumeru::GpuShaderStruct>::STRUCT_NAME,
                                type_id: ::sumeru::ComponentTypeId::of::<#comp_ty>(),
                                fields: <#comp_ty as ::sumeru::GpuShaderStruct>::FIELDS,
                                msl_def: <#comp_ty as ::sumeru::GpuShaderStruct>::msl_struct_definition,
                                wgsl_def: <#comp_ty as ::sumeru::GpuShaderStruct>::wgsl_struct_definition,
                            }
                        },
                    )
                }
            }
            other_ty => {
                if let Some((elem_kind, len)) = parse_array_type_info(other_ty) {
                    (
                        is_mut_binding,
                        false,
                        quote! {
                            ::sumeru::GpuHelperParamKind::Array {
                                elem: #elem_kind,
                                len: #len,
                            }
                        },
                    )
                } else {
                    let sk = parse_scalar_kind_tokens(other_ty)?;
                    (
                        is_mut_binding,
                        false,
                        quote! { ::sumeru::GpuHelperParamKind::Scalar(#sk) },
                    )
                }
            }
        };

        let spec_tokens = quote! {
            ::sumeru::GpuHelperParamSpec {
                name: #pname,
                is_mut: #is_mut_in_shader,
                by_mut_ref: #by_mut_ref,
                kind: #kind_tokens,
            }
        };

        params.push(ParsedHelperParam {
            ident: pat_ident.ident.clone(),
            is_mut_binding,
            ty: (*pat_type.ty).clone(),
            spec_tokens,
        });
    }

    if params.is_empty() || params.len() > 8 {
        return Err(Error::new_spanned(
            &sig.inputs,
            "`#[gpu]` helper functions must have between 1 and 8 parameters",
        ));
    }

    let root_names: Vec<String> = params.iter().map(|p| p.ident.to_string()).collect();
    let mut ir_builder = IrAstBuilder::new(root_names, has_return_val);
    let ir_block = if has_return_val {
        ir_builder.lower_block_with_tail_return(&item_fn.block)?
    } else {
        ir_builder.lower_block(&item_fn.block)?
    };

    let attrs = &item_fn.attrs;
    let vis = &item_fn.vis;
    let fn_name = &sig.ident;
    let stmts = &item_fn.block.stmts;

    let param_tokens = params.iter().map(|p| {
        let ident = &p.ident;
        let mode = if p.is_mut_binding {
            Ident::new("mut", ident.span())
        } else {
            Ident::new("ref", ident.span())
        };
        let ty = &p.ty;
        let spec = &p.spec_tokens;
        quote! { #ident : [#mode] #ty => #spec }
    });

    Ok(quote! {
        ::sumeru::sumeru_gpu_helper_fn! {
            #(#attrs)*
            #vis fn #fn_name ( #(#param_tokens),* ) -> #ret_ty_tokens => #ret_kind_opt_tokens,
            ir = #ir_block,
            body = { #(#stmts)* }
        }
    })
}

fn parse_scalar_kind_tokens(ty: &Type) -> syn::Result<TokenStream2> {
    let s = ty.to_token_stream().to_string();
    match s.trim() {
        "f32" => Ok(quote! { ::sumeru::GpuScalarKind::F32 }),
        "i32" => Ok(quote! { ::sumeru::GpuScalarKind::I32 }),
        "u32" | "usize" => Ok(quote! { ::sumeru::GpuScalarKind::U32 }),
        "bool" => Ok(quote! { ::sumeru::GpuScalarKind::Bool }),
        _ => Err(Error::new_spanned(
            ty,
            format!(
                "type `{s}` is not supported in `#[gpu]` scalar annotations/casts/uniforms; use `f32`, `i32`, `u32`, `usize`, or `bool`"
            ),
        )),
    }
}

fn parse_value_type_tokens(ty: &Type) -> syn::Result<TokenStream2> {
    if let Some((elem_kind, len)) = parse_array_type_info(ty) {
        return Ok(quote! { ::sumeru::GpuValueType::Array(#elem_kind, #len) });
    }
    let sk = parse_scalar_kind_tokens(ty)?;
    Ok(quote! { ::sumeru::GpuValueType::Scalar(#sk) })
}

/// Lexical-scope-aware AST-to-Typed-Shader-IR builder.
struct IrAstBuilder {
    scopes: Vec<Vec<String>>,
    is_helper_fn: bool,
}

impl IrAstBuilder {
    fn new(root_params: Vec<String>, is_helper_fn: bool) -> Self {
        Self {
            scopes: vec![root_params],
            is_helper_fn,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_local(&mut self, name: String) {
        if let Some(top) = self.scopes.last_mut() {
            top.push(name);
        }
    }

    fn is_in_local_scope(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|s| s.iter().any(|v| v == name))
    }

    fn lower_block(&mut self, block: &Block) -> syn::Result<TokenStream2> {
        self.push_scope();
        let mut stmt_tokens = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmt_tokens.push(self.lower_stmt(stmt)?);
        }
        self.pop_scope();
        Ok(quote! {
            ::sumeru::GpuBlock::new(::std::vec![ #(#stmt_tokens),* ])
        })
    }

    fn lower_block_with_tail_return(&mut self, block: &Block) -> syn::Result<TokenStream2> {
        self.push_scope();
        let mut stmt_tokens = Vec::with_capacity(block.stmts.len());
        let len = block.stmts.len();
        for (idx, stmt) in block.stmts.iter().enumerate() {
            if idx + 1 == len {
                stmt_tokens.push(self.lower_tail_stmt(stmt)?);
            } else {
                stmt_tokens.push(self.lower_stmt(stmt)?);
            }
        }
        self.pop_scope();
        Ok(quote! {
            ::sumeru::GpuBlock::new(::std::vec![ #(#stmt_tokens),* ])
        })
    }

    fn lower_tail_stmt(&mut self, stmt: &Stmt) -> syn::Result<TokenStream2> {
        match stmt {
            Stmt::Expr(expr, None) => self.lower_tail_expr(expr),
            _ => self.lower_stmt(stmt),
        }
    }

    fn lower_tail_expr(&mut self, expr: &Expr) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Return(_) => self.lower_stmt_expr(expr, true),
            Expr::If(expr_if) => {
                let cond = self.lower_expr(&expr_if.cond)?;
                let then_block = self.lower_block_with_tail_return(&expr_if.then_branch)?;
                let else_tokens = if let Some((_, else_expr)) = &expr_if.else_branch {
                    match &**else_expr {
                        Expr::If(_) => {
                            let elif = self.lower_tail_expr(else_expr)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseIf(::std::boxed::Box::new(#elif))) }
                        }
                        Expr::Block(eb) => {
                            let eblock = self.lower_block_with_tail_return(&eb.block)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseBlock(#eblock)) }
                        }
                        other => {
                            return Err(Error::new_spanned(
                                other,
                                "unsupported `else` branch in `#[gpu]` function",
                            ));
                        }
                    }
                } else {
                    quote! { ::std::option::Option::None }
                };
                Ok(quote! {
                    ::sumeru::GpuStmt::If {
                        cond: #cond,
                        then_block: #then_block,
                        else_branch: #else_tokens,
                    }
                })
            }
            Expr::Block(eb) => {
                if eb.label.is_some() {
                    return Err(Error::new_spanned(
                        &eb.label,
                        "labeled blocks are not supported in `#[gpu]` functions",
                    ));
                }
                let inner = self.lower_block_with_tail_return(&eb.block)?;
                Ok(quote! { ::sumeru::GpuStmt::Block(#inner) })
            }
            other => {
                let val = self.lower_expr(other)?;
                Ok(quote! { ::sumeru::GpuStmt::Return(::std::option::Option::Some(#val)) })
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> syn::Result<TokenStream2> {
        match stmt {
            Stmt::Local(local) => {
                let Some(init) = &local.init else {
                    return Err(Error::new_spanned(
                        local,
                        "`#[gpu]` local variables must be initialized at declaration",
                    ));
                };
                if init.diverge.is_some() {
                    return Err(Error::new_spanned(
                        local,
                        "`let ... else` is not supported in `#[gpu]` kernels",
                    ));
                }
                let (ident, is_mut, explicit_ty) = match &local.pat {
                    Pat::Ident(pi) => (&pi.ident, pi.mutability.is_some(), None),
                    Pat::Type(pt) => {
                        let Pat::Ident(pi) = &*pt.pat else {
                            return Err(Error::new_spanned(
                                &pt.pat,
                                "unsupported pattern in `#[gpu]` local variable declaration",
                            ));
                        };
                        let vt_tokens = parse_value_type_tokens(&pt.ty)?;
                        (&pi.ident, pi.mutability.is_some(), Some(vt_tokens))
                    }
                    other => {
                        return Err(Error::new_spanned(
                            other,
                            "destructuring patterns in `let` are not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                let vname = ident.to_string();
                if vname.starts_with("__") || vname.starts_with("sumeru_") {
                    return Err(Error::new_spanned(
                        ident,
                        "local variable name in `#[gpu]` kernel cannot start with `__` or `sumeru_`",
                    ));
                }
                let rhs = self.lower_expr(&init.expr)?;
                self.declare_local(vname.clone());
                let explicit_ty_tokens = match explicit_ty {
                    Some(k) => quote! { ::std::option::Option::Some(#k) },
                    None => quote! { ::std::option::Option::None },
                };
                Ok(quote! {
                    ::sumeru::GpuStmt::Let {
                        name: #vname,
                        mutable: #is_mut,
                        explicit_ty: #explicit_ty_tokens,
                        init: #rhs,
                    }
                })
            }
            Stmt::Item(Item::Const(item_const)) => {
                let vname = item_const.ident.to_string();
                if vname.starts_with("__") || vname.starts_with("sumeru_") {
                    return Err(Error::new_spanned(
                        &item_const.ident,
                        "const name in `#[gpu]` kernel cannot start with `__` or `sumeru_`",
                    ));
                }
                let vt_tokens = parse_value_type_tokens(&item_const.ty)?;
                let rhs = self.lower_expr(&item_const.expr)?;
                self.declare_local(vname.clone());
                Ok(quote! {
                    ::sumeru::GpuStmt::Let {
                        name: #vname,
                        mutable: false,
                        explicit_ty: ::std::option::Option::Some(#vt_tokens),
                        init: #rhs,
                    }
                })
            }
            Stmt::Item(item) => Err(Error::new_spanned(
                item,
                "nested item declarations (other than `const`) are not supported in `#[gpu]` kernels",
            )),
            Stmt::Macro(mac) => Err(Error::new_spanned(
                mac,
                "macros are not supported inside `#[gpu]` kernel bodies",
            )),
            Stmt::Expr(expr, semi) => self.lower_stmt_expr(expr, semi.is_some()),
        }
    }

    fn lower_stmt_expr(&mut self, expr: &Expr, has_semi: bool) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Assign(assign) => {
                let lhs = self.lower_lvalue(&assign.left)?;
                let rhs = self.lower_expr(&assign.right)?;
                Ok(quote! {
                    ::sumeru::GpuStmt::Assign {
                        op: ::sumeru::GpuAssignOp::Assign,
                        target: #lhs,
                        value: #rhs,
                    }
                })
            }
            Expr::Binary(bin) if is_compound_assign(&bin.op) => {
                let lhs = self.lower_lvalue(&bin.left)?;
                let op = lower_assign_op(&bin.op)?;
                let rhs = self.lower_expr(&bin.right)?;
                Ok(quote! {
                    ::sumeru::GpuStmt::Assign {
                        op: #op,
                        target: #lhs,
                        value: #rhs,
                    }
                })
            }
            Expr::Call(_) if has_semi => {
                let call_expr = self.lower_expr(expr)?;
                Ok(quote! { ::sumeru::GpuStmt::Expr(#call_expr) })
            }
            Expr::If(expr_if) => {
                let cond = self.lower_expr(&expr_if.cond)?;
                let then_block = self.lower_block(&expr_if.then_branch)?;
                let else_tokens = if let Some((_, else_expr)) = &expr_if.else_branch {
                    match &**else_expr {
                        Expr::If(_) => {
                            let else_if = self.lower_stmt_expr(else_expr, true)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseIf(::std::boxed::Box::new(#else_if))) }
                        }
                        Expr::Block(eb) => {
                            let else_body = self.lower_block(&eb.block)?;
                            quote! { ::std::option::Option::Some(::sumeru::GpuElseBranch::ElseBlock(#else_body)) }
                        }
                        other => {
                            return Err(Error::new_spanned(
                                other,
                                "unsupported `else` branch in `#[gpu]` kernel",
                            ));
                        }
                    }
                } else {
                    quote! { ::std::option::Option::None }
                };
                Ok(quote! {
                    ::sumeru::GpuStmt::If {
                        cond: #cond,
                        then_block: #then_block,
                        else_branch: #else_tokens,
                    }
                })
            }
            Expr::While(expr_while) => {
                let cond = self.lower_expr(&expr_while.cond)?;
                let body = self.lower_block(&expr_while.body)?;
                Ok(quote! {
                    ::sumeru::GpuStmt::While {
                        cond: #cond,
                        body: #body,
                    }
                })
            }
            Expr::Loop(expr_loop) => {
                if expr_loop.label.is_some() {
                    return Err(Error::new_spanned(
                        &expr_loop.label,
                        "labeled `loop` is not supported in `#[gpu]` kernels",
                    ));
                }
                let body = self.lower_block(&expr_loop.body)?;
                Ok(quote! { ::sumeru::GpuStmt::Loop(#body) })
            }
            Expr::ForLoop(expr_for) => {
                let Pat::Ident(pat_ident) = &*expr_for.pat else {
                    return Err(Error::new_spanned(
                        &expr_for.pat,
                        "`for` loop pattern in `#[gpu]` kernel must be a simple identifier",
                    ));
                };
                let var_name = pat_ident.ident.to_string();
                let Expr::Range(range) = &*expr_for.expr else {
                    return Err(Error::new_spanned(
                        &expr_for.expr,
                        "`for` loop in `#[gpu]` kernel only supports `start..end` or `start..=end` ranges",
                    ));
                };
                let (Some(start_expr), Some(end_expr)) = (&range.start, &range.end) else {
                    return Err(Error::new_spanned(
                        range,
                        "`for` loop range in `#[gpu]` kernel must specify both start and end bounds",
                    ));
                };
                let start = self.lower_expr(start_expr)?;
                let end = self.lower_expr(end_expr)?;
                let inclusive = matches!(range.limits, RangeLimits::Closed(_));
                self.push_scope();
                self.declare_local(var_name.clone());
                let body = self.lower_block(&expr_for.body)?;
                self.pop_scope();
                Ok(quote! {
                    ::sumeru::GpuStmt::ForRange {
                        var: #var_name,
                        start: #start,
                        end: #end,
                        inclusive: #inclusive,
                        body: #body,
                    }
                })
            }
            Expr::Block(eb) => {
                if eb.label.is_some() {
                    return Err(Error::new_spanned(
                        &eb.label,
                        "labeled blocks are not supported in `#[gpu]` kernels",
                    ));
                }
                let inner = self.lower_block(&eb.block)?;
                Ok(quote! { ::sumeru::GpuStmt::Block(#inner) })
            }
            Expr::Return(ret) => {
                if let Some(val) = &ret.expr {
                    if !self.is_helper_fn {
                        return Err(Error::new_spanned(
                            val,
                            "`#[gpu]` kernel cannot return a value",
                        ));
                    }
                    let lowered_val = self.lower_expr(val)?;
                    Ok(quote! { ::sumeru::GpuStmt::Return(::std::option::Option::Some(#lowered_val)) })
                } else {
                    if self.is_helper_fn {
                        return Err(Error::new_spanned(
                            ret,
                            "`#[gpu]` helper function with a return type must return a value",
                        ));
                    }
                    Ok(quote! { ::sumeru::GpuStmt::Return(::std::option::Option::None) })
                }
            }
            Expr::Break(brk) => {
                if brk.label.is_some() || brk.expr.is_some() {
                    return Err(Error::new_spanned(
                        brk,
                        "labeled or value `break` is not supported in `#[gpu]` kernels",
                    ));
                }
                Ok(quote! { ::sumeru::GpuStmt::Break })
            }
            Expr::Continue(cont) => {
                if cont.label.is_some() {
                    return Err(Error::new_spanned(
                        cont,
                        "labeled `continue` is not supported in `#[gpu]` kernels",
                    ));
                }
                Ok(quote! { ::sumeru::GpuStmt::Continue })
            }
            other => {
                if !has_semi {
                    return Err(Error::new_spanned(
                        other,
                        "`#[gpu]` kernel cannot end with an implicit return expression; add a semicolon or use an assignment",
                    ));
                }
                Err(Error::new_spanned(
                    other,
                    "unsupported statement in `#[gpu]` kernel; only assignments, local `let`, helper calls, `if`/`else`, `while`, `loop`, `for`, and `return` are supported",
                ))
            }
        }
    }

    fn lower_lvalue(&mut self, expr: &Expr) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {
                let name = p.path.segments[0].ident.to_string();
                Ok(quote! { ::sumeru::GpuLValue::Var(#name) })
            }
            Expr::Field(f) => {
                let base = self.lower_lvalue(&f.base)?;
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => {
                        return Err(Error::new_spanned(
                            idx,
                            "tuple field access is not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                Ok(quote! {
                    ::sumeru::GpuLValue::Field {
                        base: ::std::boxed::Box::new(#base),
                        field: #member,
                    }
                })
            }
            Expr::Index(idx) => {
                let base = self.lower_lvalue(&idx.expr)?;
                let index = self.lower_expr(&idx.index)?;
                Ok(quote! {
                    ::sumeru::GpuLValue::Index {
                        base: ::std::boxed::Box::new(#base),
                        index: ::std::boxed::Box::new(#index),
                    }
                })
            }
            Expr::Unary(un) if matches!(un.op, UnOp::Deref(_)) => self.lower_lvalue(&un.expr),
            Expr::Paren(p) => self.lower_lvalue(&p.expr),
            other => Err(Error::new_spanned(
                other,
                "invalid assignment target in `#[gpu]` kernel",
            )),
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> syn::Result<TokenStream2> {
        match expr {
            Expr::Path(p) => {
                if p.qself.is_none() && p.path.segments.len() == 1 {
                    let ident_str = p.path.segments[0].ident.to_string();
                    if self.is_in_local_scope(&ident_str) {
                        return Ok(quote! { ::sumeru::GpuExpr::Var(#ident_str) });
                    }
                }
                // External Rust `const` identifier or multi-segment path (e.g. `DT` or `std::f32::consts::PI`):
                // evaluated at compile time by `rustc` via `GpuConstValue`.
                Ok(quote! { ::sumeru::GpuConstValue::to_gpu_expr(&(#p)) })
            }
            Expr::Reference(expr_ref) => self.lower_expr(&expr_ref.expr),
            Expr::Array(expr_arr) => {
                if !(2..=4).contains(&expr_arr.elems.len()) {
                    return Err(Error::new_spanned(
                        expr_arr,
                        "fixed-size array literals in `#[gpu]` functions must have between 2 and 4 elements",
                    ));
                }
                let mut elems = Vec::with_capacity(expr_arr.elems.len());
                for e in &expr_arr.elems {
                    elems.push(self.lower_expr(e)?);
                }
                Ok(quote! { ::sumeru::GpuExpr::ArrayInit(::std::vec![ #(#elems),* ]) })
            }
            Expr::Repeat(expr_rep) => {
                let Expr::Lit(el) = &*expr_rep.len else {
                    return Err(Error::new_spanned(
                        &expr_rep.len,
                        "array repeat length in `#[gpu]` functions must be an integer literal (2..=4)",
                    ));
                };
                let Lit::Int(il) = &el.lit else {
                    return Err(Error::new_spanned(
                        &expr_rep.len,
                        "array repeat length in `#[gpu]` functions must be an integer literal (2..=4)",
                    ));
                };
                let len: usize = il.base10_parse()?;
                if !(2..=4).contains(&len) {
                    return Err(Error::new_spanned(
                        expr_rep,
                        "array repeat length in `#[gpu]` functions must be between 2 and 4",
                    ));
                }
                let elem = self.lower_expr(&expr_rep.expr)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::ArrayRepeat {
                        elem: ::std::boxed::Box::new(#elem),
                        len: #len,
                    }
                })
            }
            Expr::Field(f) => {
                let base = self.lower_expr(&f.base)?;
                let member = match &f.member {
                    Member::Named(id) => id.to_string(),
                    Member::Unnamed(idx) => {
                        return Err(Error::new_spanned(
                            idx,
                            "tuple field access is not supported in `#[gpu]` kernels",
                        ));
                    }
                };
                Ok(quote! {
                    ::sumeru::GpuExpr::Field {
                        base: ::std::boxed::Box::new(#base),
                        field: #member,
                    }
                })
            }
            Expr::Index(idx) => {
                let base = self.lower_expr(&idx.expr)?;
                let index = self.lower_expr(&idx.index)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Index {
                        base: ::std::boxed::Box::new(#base),
                        index: ::std::boxed::Box::new(#index),
                    }
                })
            }
            Expr::Binary(bin) => {
                if is_compound_assign(&bin.op) {
                    return Err(Error::new_spanned(
                        bin,
                        "compound assignment cannot be used as a sub-expression in `#[gpu]` kernels",
                    ));
                }
                let lhs = self.lower_expr(&bin.left)?;
                let op = lower_binop(&bin.op)?;
                let rhs = self.lower_expr(&bin.right)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Binary {
                        op: #op,
                        lhs: ::std::boxed::Box::new(#lhs),
                        rhs: ::std::boxed::Box::new(#rhs),
                    }
                })
            }
            Expr::Unary(un) => match un.op {
                UnOp::Neg(_) => {
                    let inner = self.lower_expr(&un.expr)?;
                    Ok(quote! {
                        ::sumeru::GpuExpr::Unary {
                            op: ::sumeru::GpuUnaryOp::Neg,
                            expr: ::std::boxed::Box::new(#inner),
                        }
                    })
                }
                UnOp::Not(_) => {
                    let inner = self.lower_expr(&un.expr)?;
                    Ok(quote! {
                        ::sumeru::GpuExpr::Unary {
                            op: ::sumeru::GpuUnaryOp::Not,
                            expr: ::std::boxed::Box::new(#inner),
                        }
                    })
                }
                UnOp::Deref(_) => self.lower_expr(&un.expr),
                _ => Err(Error::new_spanned(
                    un,
                    "unsupported unary operator in `#[gpu]` kernel",
                )),
            },
            Expr::Paren(p) => {
                let inner = self.lower_expr(&p.expr)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Paren(::std::boxed::Box::new(#inner))
                })
            }
            Expr::Cast(cast) => {
                let inner = self.lower_expr(&cast.expr)?;
                let ty = parse_scalar_kind_tokens(&cast.ty)?;
                Ok(quote! {
                    ::sumeru::GpuExpr::Cast {
                        expr: ::std::boxed::Box::new(#inner),
                        ty: #ty,
                    }
                })
            }
            Expr::Lit(expr_lit) => self.lower_lit(&expr_lit.lit),
            Expr::MethodCall(mc) => {
                let receiver = self.lower_expr(&mc.receiver)?;
                let method = mc.method.to_string();
                let mut args = Vec::with_capacity(mc.args.len());
                for arg in &mc.args {
                    args.push(self.lower_expr(arg)?);
                }
                let intrinsic = match (method.as_str(), args.len()) {
                    ("abs", 0) => quote! { ::sumeru::GpuMathIntrinsic::Abs },
                    ("sqrt", 0) => quote! { ::sumeru::GpuMathIntrinsic::Sqrt },
                    ("sin", 0) => quote! { ::sumeru::GpuMathIntrinsic::Sin },
                    ("cos", 0) => quote! { ::sumeru::GpuMathIntrinsic::Cos },
                    ("tan", 0) => quote! { ::sumeru::GpuMathIntrinsic::Tan },
                    ("floor", 0) => quote! { ::sumeru::GpuMathIntrinsic::Floor },
                    ("ceil", 0) => quote! { ::sumeru::GpuMathIntrinsic::Ceil },
                    ("round", 0) => quote! { ::sumeru::GpuMathIntrinsic::Round },
                    ("fract", 0) => quote! { ::sumeru::GpuMathIntrinsic::Fract },
                    ("exp", 0) => quote! { ::sumeru::GpuMathIntrinsic::Exp },
                    ("ln", 0) => quote! { ::sumeru::GpuMathIntrinsic::Ln },
                    ("min", 1) => quote! { ::sumeru::GpuMathIntrinsic::Min },
                    ("max", 1) => quote! { ::sumeru::GpuMathIntrinsic::Max },
                    ("atan2", 1) => quote! { ::sumeru::GpuMathIntrinsic::Atan2 },
                    ("powf", 1) => quote! { ::sumeru::GpuMathIntrinsic::Powf },
                    ("clamp", 2) => quote! { ::sumeru::GpuMathIntrinsic::Clamp },
                    ("mul_add" | "fma", 2) => quote! { ::sumeru::GpuMathIntrinsic::Fma },
                    ("dot", 1) => quote! { ::sumeru::GpuMathIntrinsic::Dot },
                    ("length", 0) => quote! { ::sumeru::GpuMathIntrinsic::Length },
                    ("length_squared", 0) => {
                        return Ok(quote! {
                            ::sumeru::GpuExpr::MathCall {
                                receiver: ::std::boxed::Box::new(#receiver),
                                intrinsic: ::sumeru::GpuMathIntrinsic::Dot,
                                args: ::std::vec![ #receiver ],
                            }
                        });
                    }
                    ("normalize", 0) => quote! { ::sumeru::GpuMathIntrinsic::Normalize },
                    ("distance", 1) => quote! { ::sumeru::GpuMathIntrinsic::Distance },
                    _ => {
                        return Err(Error::new_spanned(
                            &mc.method,
                            format!(
                                "method `{method}` (with {} arg(s)) is not supported in `#[gpu]` kernels; supported methods: abs, sqrt, min, max, clamp, sin, cos, tan, floor, ceil, round, fract, exp, ln, powf, atan2, mul_add, dot, length, length_squared, normalize, distance",
                                args.len()
                            ),
                        ));
                    }
                };
                Ok(quote! {
                    ::sumeru::GpuExpr::MathCall {
                        receiver: ::std::boxed::Box::new(#receiver),
                        intrinsic: #intrinsic,
                        args: ::std::vec![ #(#args),* ],
                    }
                })
            }
            Expr::Struct(st) => {
                let type_name = st
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let expected: &[&str] = match type_name.as_str() {
                    "Vec2" => &["x", "y"],
                    "Vec3" => &["x", "y", "z"],
                    "Vec4" => &["x", "y", "z", "w"],
                    _ => {
                        return Err(Error::new_spanned(
                            st,
                            "only `Vec2`, `Vec3`, and `Vec4` struct literals are supported in `#[gpu]` / `#[kernel]` functions",
                        ));
                    }
                };
                if st.rest.is_some() {
                    return Err(Error::new_spanned(
                        st,
                        "struct update syntax (`..rest`) is not supported in `#[gpu]` / `#[kernel]` functions",
                    ));
                }
                let mut elems = Vec::with_capacity(expected.len());
                for &comp in expected {
                    let Some(fv) = st.fields.iter().find(|f| matches!(&f.member, Member::Named(id) if id == comp)) else {
                        return Err(Error::new_spanned(
                            st,
                            format!("missing field `{comp}` in `{type_name}` literal"),
                        ));
                    };
                    elems.push(self.lower_expr(&fv.expr)?);
                }
                Ok(quote! {
                    ::sumeru::GpuExpr::VecConstruct {
                        elem_ty: ::sumeru::GpuScalarKind::F32,
                        elements: ::std::vec![ #(#elems),* ],
                    }
                })
            }
            Expr::Call(call) => {
                let Expr::Path(func_path) = &*call.func else {
                    return Err(Error::new_spanned(
                        &call.func,
                        "only direct calls to `#[gpu]` helper functions are supported in `#[gpu]` kernels",
                    ));
                };
                if func_path.qself.is_some() {
                    return Err(Error::new_spanned(
                        func_path,
                        "qualified `<T as Trait>::fn` calls are not supported in `#[gpu]` kernels",
                    ));
                };
                let mut args = Vec::with_capacity(call.args.len());
                for arg in &call.args {
                    args.push(self.lower_expr(arg)?);
                }
                let path = &func_path.path;
                if path.segments.len() >= 2 {
                    let prev = path.segments[path.segments.len() - 2].ident.to_string();
                    let last = path.segments.last().unwrap().ident.to_string();
                    if matches!(prev.as_str(), "Vec2" | "Vec3" | "Vec4") {
                        let expected_len: usize = match prev.as_str() {
                            "Vec2" => 2,
                            "Vec3" => 3,
                            _ => 4,
                        };
                        match (last.as_str(), args.len()) {
                            ("new", n) if n == expected_len => {
                                return Ok(quote! {
                                    ::sumeru::GpuExpr::VecConstruct {
                                        elem_ty: ::sumeru::GpuScalarKind::F32,
                                        elements: ::std::vec![ #(#args),* ],
                                    }
                                });
                            }
                            ("splat", 1) => {
                                let a0 = &args[0];
                                return Ok(quote! {
                                    ::sumeru::GpuExpr::ArrayRepeat {
                                        elem: ::std::boxed::Box::new(#a0),
                                        len: #expected_len,
                                    }
                                });
                            }
                            ("from_array", 1) => {
                                let a0 = &args[0];
                                return Ok(quote! { #a0 });
                            }
                            _ => {}
                        }
                    }
                }
                if path.is_ident("vec2") {
                    return Ok(quote! {
                        ::sumeru::GpuExpr::VecConstruct {
                            elem_ty: ::sumeru::GpuScalarKind::F32,
                            elements: ::std::vec![ #(#args),* ],
                        }
                    });
                }
                if path.is_ident("vec3") {
                    return Ok(quote! {
                        ::sumeru::GpuExpr::VecConstruct {
                            elem_ty: ::sumeru::GpuScalarKind::F32,
                            elements: ::std::vec![ #(#args),* ],
                        }
                    });
                }
                if path.is_ident("vec4") {
                    return Ok(quote! {
                        ::sumeru::GpuExpr::VecConstruct {
                            elem_ty: ::sumeru::GpuScalarKind::F32,
                            elements: ::std::vec![ #(#args),* ],
                        }
                    });
                }
                if path.is_ident("dot") && args.len() == 2 {
                    let a0 = &args[0];
                    let a1 = &args[1];
                    return Ok(quote! {
                        ::sumeru::GpuExpr::MathCall {
                            receiver: ::std::boxed::Box::new(#a0),
                            intrinsic: ::sumeru::GpuMathIntrinsic::Dot,
                            args: ::std::vec![ #a1 ],
                        }
                    });
                }
                if path.is_ident("length") && args.len() == 1 {
                    let a0 = &args[0];
                    return Ok(quote! {
                        ::sumeru::GpuExpr::MathCall {
                            receiver: ::std::boxed::Box::new(#a0),
                            intrinsic: ::sumeru::GpuMathIntrinsic::Length,
                            args: ::std::vec![],
                        }
                    });
                }
                if path.is_ident("normalize") && args.len() == 1 {
                    let a0 = &args[0];
                    return Ok(quote! {
                        ::sumeru::GpuExpr::MathCall {
                            receiver: ::std::boxed::Box::new(#a0),
                            intrinsic: ::sumeru::GpuMathIntrinsic::Normalize,
                            args: ::std::vec![],
                        }
                    });
                }
                if path.is_ident("distance") && args.len() == 2 {
                    let a0 = &args[0];
                    let a1 = &args[1];
                    return Ok(quote! {
                        ::sumeru::GpuExpr::MathCall {
                            receiver: ::std::boxed::Box::new(#a0),
                            intrinsic: ::sumeru::GpuMathIntrinsic::Distance,
                            args: ::std::vec![ #a1 ],
                        }
                    });
                }
                Ok(quote! {
                    ::sumeru::GpuExpr::HelperCall {
                        helper: #path::gpu_helper_descriptor(),
                        args: ::std::vec![ #(#args),* ],
                    }
                })
            }
            other => Err(Error::new_spanned(
                other,
                "unsupported Rust expression in `#[gpu]` kernel; closures and `match` are not permitted in GPU shaders",
            )),
        }
    }

    fn lower_lit(&self, lit: &Lit) -> syn::Result<TokenStream2> {
        match lit {
            Lit::Float(fl) => {
                let digits = fl.base10_digits().replace('_', "");
                let formatted = if digits.contains('.') || digits.contains('e') || digits.contains('E') {
                    digits
                } else {
                    format!("{digits}.0")
                };
                Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::FloatRaw(#formatted)) })
            }
            Lit::Int(il) => {
                let suffix = il.suffix();
                match suffix {
                    "" => {
                        let val: i64 = il.base10_parse()?;
                        Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::UntypedInt(#val)) })
                    }
                    "i32" => {
                        let val: i32 = il.base10_parse()?;
                        Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::I32(#val)) })
                    }
                    "u32" | "usize" => {
                        let val: u64 = il.base10_parse()?;
                        Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::U32(#val)) })
                    }
                    other => Err(Error::new_spanned(
                        il,
                        format!(
                            "integer literal suffix `{other}` is not supported in `#[gpu]` kernels; use `i32`, `u32`, or `usize`"
                        ),
                    )),
                }
            }
            Lit::Bool(b) => {
                let val = b.value;
                Ok(quote! { ::sumeru::GpuExpr::Lit(::sumeru::GpuLit::Bool(#val)) })
            }
            other => Err(Error::new_spanned(
                other,
                "string/byte/char literals are not supported in `#[gpu]` kernels",
            )),
        }
    }
}

fn is_compound_assign(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

fn lower_assign_op(op: &BinOp) -> syn::Result<TokenStream2> {
    match op {
        BinOp::AddAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::AddAssign }),
        BinOp::SubAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::SubAssign }),
        BinOp::MulAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::MulAssign }),
        BinOp::DivAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::DivAssign }),
        BinOp::RemAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::RemAssign }),
        BinOp::BitXorAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::BitXorAssign }),
        BinOp::BitAndAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::BitAndAssign }),
        BinOp::BitOrAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::BitOrAssign }),
        BinOp::ShlAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::ShlAssign }),
        BinOp::ShrAssign(_) => Ok(quote! { ::sumeru::GpuAssignOp::ShrAssign }),
        other => Err(Error::new_spanned(
            other,
            "unsupported compound assignment operator in `#[gpu]` kernel",
        )),
    }
}

fn lower_binop(op: &BinOp) -> syn::Result<TokenStream2> {
    match op {
        BinOp::Add(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Add }),
        BinOp::Sub(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Sub }),
        BinOp::Mul(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Mul }),
        BinOp::Div(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Div }),
        BinOp::Rem(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Rem }),
        BinOp::And(_) => Ok(quote! { ::sumeru::GpuBinaryOp::And }),
        BinOp::Or(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Or }),
        BinOp::BitXor(_) => Ok(quote! { ::sumeru::GpuBinaryOp::BitXor }),
        BinOp::BitAnd(_) => Ok(quote! { ::sumeru::GpuBinaryOp::BitAnd }),
        BinOp::BitOr(_) => Ok(quote! { ::sumeru::GpuBinaryOp::BitOr }),
        BinOp::Shl(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Shl }),
        BinOp::Shr(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Shr }),
        BinOp::Eq(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Eq }),
        BinOp::Lt(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Lt }),
        BinOp::Le(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Le }),
        BinOp::Ne(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Ne }),
        BinOp::Ge(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Ge }),
        BinOp::Gt(_) => Ok(quote! { ::sumeru::GpuBinaryOp::Gt }),
        other => Err(Error::new_spanned(
            other,
            "unsupported binary operator in `#[gpu]` kernel",
        )),
    }
}
