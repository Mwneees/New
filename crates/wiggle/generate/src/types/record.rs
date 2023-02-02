use crate::lifetimes::{anon_lifetime, LifetimeExt};
use crate::names;

use proc_macro2::TokenStream;
use quote::quote;
use witx::Layout;

pub(super) fn define_struct(name: &witx::Id, s: &witx::RecordDatatype) -> TokenStream {
    let ident = names::type_(name);
    let size = s.mem_size_align().size as u32;
    let align = s.mem_size_align().align as usize;

    let member_names = s.members.iter().map(|m| names::struct_member(&m.name));
    let member_decls = s.members.iter().map(|m| {
        let name = names::struct_member(&m.name);
        let type_ = match &m.tref {
            witx::TypeRef::Name(nt) => {
                let tt = names::type_(&nt.name);
                if m.tref.needs_lifetime() {
                    quote!(#tt<'a>)
                } else {
                    quote!(#tt)
                }
            }
            witx::TypeRef::Value(ty) => match &**ty {
                witx::Type::Builtin(builtin) => names::builtin_type(*builtin),
                witx::Type::Pointer(pointee) | witx::Type::ConstPointer(pointee) => {
                    let pointee_type = names::type_ref(&pointee, quote!('a));
                    quote!(wiggle::GuestPtr<'a, #pointee_type>)
                }
                _ => unimplemented!("other anonymous struct members: {:?}", m.tref),
            },
        };
        quote!(pub #name: #type_)
    });

    let member_reads = s.member_layout().into_iter().map(|ml| {
        let name = names::struct_member(&ml.member.name);
        let offset = ml.offset as u32;
        let location = quote!(location.cast::<u8>().add(#offset)?.cast());
        match &ml.member.tref {
            witx::TypeRef::Name(nt) => {
                let type_ = names::type_(&nt.name);
                quote! {
                    let #name = <#type_ as wiggle::GuestType>::read(&#location)?;
                }
            }
            witx::TypeRef::Value(ty) => match &**ty {
                witx::Type::Builtin(builtin) => {
                    let type_ = names::builtin_type(*builtin);
                    quote! {
                        let #name = <#type_ as wiggle::GuestType>::read(&#location)?;
                    }
                }
                witx::Type::Pointer(pointee) | witx::Type::ConstPointer(pointee) => {
                    let pointee_type = names::type_ref(&pointee, anon_lifetime());
                    quote! {
                        let #name = <wiggle::GuestPtr::<#pointee_type> as wiggle::GuestType>::read(&#location)?;
                    }
                }
                _ => unimplemented!("other anonymous struct members: {:?}", ty),
            },
        }
    });

    let member_writes = s.member_layout().into_iter().map(|ml| {
        let name = names::struct_member(&ml.member.name);
        let offset = ml.offset as u32;
        quote! {
            wiggle::GuestType::write(
                &location.cast::<u8>().add(#offset)?.cast(),
                val.#name,
            )?;
        }
    });

    let (struct_lifetime, extra_derive) = if s.needs_lifetime() {
        (quote!(<'a>), quote!())
    } else {
        (quote!(), quote!(, PartialEq))
    };

    quote! {
        #[derive(Clone, Debug #extra_derive)]
        pub struct #ident #struct_lifetime {
            #(#member_decls),*
        }

        impl<'a> wiggle::GuestType<'a> for #ident #struct_lifetime {
            #[inline]
            fn guest_size() -> u32 {
                #size
            }

            #[inline]
            fn guest_align() -> usize {
                #align
            }

            fn read(location: &wiggle::GuestPtr<'a, Self>) -> Result<Self, wiggle::GuestError> {
                #(#member_reads)*
                Ok(#ident { #(#member_names),* })
            }

            fn write(location: &wiggle::GuestPtr<'_, Self>, val: Self) -> Result<(), wiggle::GuestError> {
                #(#member_writes)*
                Ok(())
            }
        }
    }
}

impl super::WiggleType for witx::RecordDatatype {
    fn impls_display(&self) -> bool {
        false
    }
}
