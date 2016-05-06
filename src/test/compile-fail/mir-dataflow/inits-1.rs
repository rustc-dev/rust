// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

struct S(i32);

// Here is the more usual way you would write this test case.  (I have
// added numerous line breaks to try to clarify how the expressions
// are decomposed into the MIR that is reflected in the dataflow
// analysis results.)
#[cfg(simplified)]
fn foo(test: bool, x: &mut S, y: S, mut z: S) -> S {
    let ret;
    ret = if test {
        ::std::mem::replace(x, y)
    } else {
        z = y;
        z
    };
    ::std::mem::drop(x);
    ret
}


#[rustc_mir_borrowck]
#[rustc_mir(borrowck_graphviz_postflow="/tmp/inits-1.dot",dataflow_info_maybe_init)]
//~                  ERROR           entry dataflow:maybe_init [arg0=test|arg1=x|arg2=y|arg3=z]
fn foo(test: bool, x: &mut S, y: S, mut z: S) -> S {

    // (1. At the outset, all parameters are considered implicitly
    //     initialized; The first maps to arg0, the second to arg1, et
    //     cetera. On entry to the function, each arg is moved, or
    //     copied if appropriate, into a corresponding var.)

    let ret;
    let other;

    // (2. For an assignment, we evaluate the right-hand side (RHS)
    //     first into a temp before moving it into the destination.)
    //
    //     This entry line reflects the dataflow state *after*
    //     evaluating the RHS.)

    //~              ERROR           entry dataflow:maybe_init [var0=test|var1=x|var3=z|var5=other|arg0=test|tmp0]
    ret =

        // (3. expression is evaluated into tmp0)
        if test {
            //~^     ERROR exit:if         dataflow:maybe_init [var0=test|var1=x|var2=y|var3=z|arg0=test]

            //~      ERROR           entry dataflow:maybe_init [var0=test|var1=x|var2=y|var3=z|arg0=test]
            ::std::mem::replace(x, y)
                //~^ ERROR exit:call       dataflow:maybe_init [var0=test|var1=x|var3=z|arg0=test]

        } else {
            //~      ERROR           entry dataflow:maybe_init [var0=test|var1=x|var2=y|var3=z|arg0=test]
            other = y;
            z
        }
    //~^             ERROR exit:goto       dataflow:maybe_init [var0=test|var1=x|var5=other|arg0=test|tmp0]

    // (4. Then tmp0 is moved into var4, aka `ret` in the program.
    //
    //     This is the start of a new basic block; its associated
    //     entry dataflow state is noted below annotation (2) above.)
    ;

    // (5. Continuing the basic block, the `()` result of this call is
    //     evaluated into tmp6)
    ::std::mem::drop(x)
        //~^         ERROR exit:call       dataflow:maybe_init [var0=test|var3=z|var4=ret|var5=other|arg0=test]

        // (6. tmp6 is not moved into any binding, but since `()` is
        //     `Copy`, it remains alive.)
        ;

    //~              ERROR           entry dataflow:maybe_init [var0=test|var3=z|var4=ret|var5=other|arg0=test|tmp6]
    ret

        // (7. there are two basic blocks that end up having spans
        //     that end at the same place; the first block initalizes
        //     the return L-value and then jumps, via goto, the block
        //     with a return statement.)
}
//~^                 ERROR exit:goto       dataflow:maybe_init [var0=test|var3=z|var5=other|arg0=test|tmp6|return]
//~|                 ERROR           entry dataflow:maybe_init [var0=test|var3=z|var5=other|arg0=test|tmp6|return]
//~|                 ERROR exit:return     dataflow:maybe_init [var0=test|var3=z|var5=other|arg0=test|tmp6]

fn main() {
    foo(true, &mut S(13), S(14), S(15));
    foo(false, &mut S(13), S(14), S(15));
}
