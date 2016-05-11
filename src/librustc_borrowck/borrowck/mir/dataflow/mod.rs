// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::attr::AttrMetaMethods;
use syntax::codemap::{DUMMY_SP};

use rustc::ty::TyCtxt;
use rustc::mir::repr::{self, Mir};

use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::path::PathBuf;
use std::usize;

use super::MirBorrowckCtxtPreDataflow;
use super::gather_moves::{Location, MoveData, MovePathData, MovePathIndex, MoveOutIndex, PathMap};
use super::gather_moves::{MoveOut, MovePath, MovePathContent};

use bitslice::BitSlice; // adds set_bit/get_bit to &[usize] bitvector rep.

pub use self::info_warn::issue_result_info;

mod graphviz;
mod info_warn;

pub trait Dataflow {
    fn dataflow(&mut self);
}

impl<'b, 'a: 'b, 'tcx: 'a, BD> Dataflow for MirBorrowckCtxtPreDataflow<'b, 'a, 'tcx, BD>
    where BD: BitDenotation, BD::Bit: Debug, BD::Ctxt: HasMoveData<'tcx>
{
    fn dataflow(&mut self) {
        self.flow_state.build_sets();
        self.pre_dataflow_instrumentation().unwrap();
        self.flow_state.propagate();
        self.post_dataflow_instrumentation().unwrap();
    }
}

struct PropagationContext<'b, 'a: 'b, 'tcx: 'a, O>
    where O: 'b + BitDenotation, O::Ctxt: HasMoveData<'tcx>,
{
    builder: &'b mut DataflowAnalysis<'a, 'tcx, O>,
    changed: bool,
}

impl<'a, 'tcx: 'a, BD> DataflowAnalysis<'a, 'tcx, BD>
    where BD: BitDenotation, BD::Ctxt: HasMoveData<'tcx>
{
    fn propagate(&mut self) {
        let mut temp = vec![0; self.flow_state.sets.words_per_block];
        let mut propcx = PropagationContext {
            builder: self,
            changed: true,
        };
        while propcx.changed {
            propcx.changed = false;
            propcx.reset(&mut temp);
            propcx.walk_cfg(&mut temp);
        }
    }

    fn build_sets(&mut self) {
        // First we need to build the entry-, gen- and kill-sets. The
        // gather_moves information provides a high-level mapping from
        // mir-locations to the MoveOuts (and those correspond
        // directly to gen-sets here). But we still need to figure out
        // the kill-sets.

        let move_data = self.ctxt.move_data();

        {
            let sets = &mut self.flow_state.sets.for_block(repr::START_BLOCK.index());
            self.flow_state.operator.start_block_effect(self.tcx, &self.ctxt, self.mir, sets);
        }

        for bb in self.mir.all_basic_blocks() {
            let &repr::BasicBlockData { ref statements,
                                        ref terminator,
                                        is_cleanup: _ } =
                self.mir.basic_block_data(bb);

            let sets = &mut self.flow_state.sets.for_block(bb.index());
            for j_stmt in statements.iter().enumerate() {
                self.flow_state.operator.statement_effect(self.tcx, &self.ctxt, sets, bb, j_stmt);
            }

            if let Some(ref term) = *terminator {
                let stmts_len = statements.len();
                self.flow_state.operator.terminator_effect(self.tcx, &self.ctxt, sets, bb, (stmts_len, term));
            }
        }
    }
}

fn on_all_children_bits<Each>(set: &mut [usize],
                              path_map: &PathMap,
                              move_paths: &MovePathData,
                              move_path_index: MovePathIndex,
                              each_child: &Each)
    where Each: Fn(&mut [usize], MoveOutIndex)
{
    // 1. invoke `each_child` callback for all moves that directly
    //    influence path for `move_path_index`
    for move_index in &path_map[move_path_index] {
        each_child(set, *move_index);
    }

    // 2. for each child of the path (that is named in this
    //    function), recur.
    //
    // (Unnamed children are irrelevant to dataflow; by
    // definition they have no associated moves.)
    let mut next_child_index = move_paths[move_path_index].first_child;
    while let Some(child_index) = next_child_index {
        on_all_children_bits(set, path_map, move_paths, child_index, each_child);
        next_child_index = move_paths[child_index].next_sibling;
    }
}

impl<'b, 'a: 'b, 'tcx: 'a, BD> PropagationContext<'b, 'a, 'tcx, BD>
    where BD: BitDenotation, BD::Ctxt: HasMoveData<'tcx>
{
    fn reset(&mut self, bits: &mut [usize]) {
        let e = if BD::initial_value() {usize::MAX} else {0};
        for b in bits {
            *b = e;
        }
    }

    fn walk_cfg(&mut self, in_out: &mut [usize]) {
        let mir = self.builder.mir;
        for (bb_idx, bb_data) in mir.basic_blocks.iter().enumerate() {
            let builder = &mut self.builder;
            {
                let sets = builder.flow_state.sets.for_block(bb_idx);
                debug_assert!(in_out.len() == sets.on_entry.len());
                in_out.clone_from_slice(sets.on_entry);
                bitwise(in_out, sets.gen_set, &Union);
                bitwise(in_out, sets.kill_set, &Subtract);
            }
            builder.propagate_bits_into_graph_successors_of(in_out,
                                                            &mut self.changed,
                                                            (repr::BasicBlock::new(bb_idx), bb_data));
        }
    }
}

impl<'b, 'a: 'b, 'tcx: 'a, BD> MirBorrowckCtxtPreDataflow<'b, 'a, 'tcx, BD>
    where BD: BitDenotation, BD::Bit: Debug, BD::Ctxt: HasMoveData<'tcx>
{
    fn path(context: &str, prepost: &str, path: &str) -> PathBuf {
        format!("{}_{}", context, prepost);
        let mut path = PathBuf::from(path);
        let new_file_name = {
            let orig_file_name = path.file_name().unwrap().to_str().unwrap();
            format!("{}_{}", context, orig_file_name)
        };
        path.set_file_name(new_file_name);
        path
    }

    fn pre_dataflow_instrumentation(&self) -> io::Result<()> {
        if let Some(ref path_str) = self.print_preflow_to {
            let path = Self::path(BD::name(), "preflow", path_str);
            graphviz::print_borrowck_graph_to(self, &path)
        } else {
            Ok(())
        }
    }

    fn post_dataflow_instrumentation(&self) -> io::Result<()> {
        if let Some(ref path_str) = self.print_postflow_to {
            let path = Self::path(BD::name(), "postflow", path_str);
            graphviz::print_borrowck_graph_to(self, &path)
        } else{
            Ok(())
        }
    }
}

/// Maps each block to a set of bits
#[derive(Clone, Debug)]
struct Bits {
    bits: Vec<usize>,
}

impl Bits {
    fn new(init_word: usize, num_words: usize) -> Self {
        Bits { bits: vec![init_word; num_words] }
    }
}

pub trait HasMoveData<'tcx> {
    fn move_data(&self) -> &MoveData<'tcx>;
}

impl<'tcx> HasMoveData<'tcx> for MoveData<'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { self }
}
impl<'tcx, A, B> HasMoveData<'tcx> for (A, B, MoveData<'tcx>) {
    fn move_data(&self) -> &MoveData<'tcx> { &self.2 }
}

pub struct DataflowAnalysis<'a, 'tcx: 'a, O>
    where O: BitDenotation, O::Ctxt: HasMoveData<'tcx>
{
    flow_state: DataflowState<O>,
    ctxt: O::Ctxt,
    mir: &'a Mir<'tcx>,
    tcx: &'a TyCtxt<'tcx>,
}

impl<'a, 'tcx: 'a, O> DataflowAnalysis<'a, 'tcx, O>
    where O: BitDenotation, O::Ctxt: HasMoveData<'tcx>
{
    pub fn results(self) -> (O::Ctxt, DataflowResults<O>) {
        (self.ctxt, DataflowResults(self.flow_state))
    }
}

#[derive(Debug)]
pub struct DataflowResults<O: BitDenotation>(DataflowState<O>);

impl<O: BitDenotation> DataflowResults<O> {
    fn sets(&self) -> &AllSets {
        &self.0.sets
    }
}

#[derive(Debug)]
struct DataflowState<O: BitDenotation>
{
    /// All the sets for the analysis. (Factored into its
    /// own structure so that we can borrow it mutably
    /// on its own separate from other fields.)
    pub sets: AllSets,

    /// operator used to initialize, combine, and interpret bits.
    operator: O,
}

#[derive(Debug)]
pub struct AllSets {
    /// Analysis bitwidth for each block.
    bits_per_block: usize,

    /// Number of words associated with each block entry
    /// equal to bits_per_block / usize::BITS, rounded up.
    words_per_block: usize,

    /// For each block, bits generated by executing the statements in
    /// the block. (For comparison, the Terminator for each block is
    /// handled in a flow-specific manner during propagation.)
    gen_sets: Bits,

    /// For each block, bits killed by executing the statements in the
    /// block. (For comparison, the Terminator for each block is
    /// handled in a flow-specific manner during propagation.)
    kill_sets: Bits,

    /// For each block, bits valid on entry to the block.
    on_entry_sets: Bits,
}

pub struct BlockSets<'a> {
    on_entry: &'a mut [usize],
    gen_set: &'a mut [usize],
    kill_set: &'a mut [usize],
}

impl AllSets {
    pub fn bits_per_block(&self) -> usize { self.bits_per_block }
    pub fn bytes_per_block(&self) -> usize { (self.bits_per_block + 7) / 8 }
    pub fn for_block(&mut self, block_idx: usize) -> BlockSets {
        let offset = self.words_per_block * block_idx;
        let range = offset..(offset + self.words_per_block);
        BlockSets {
            on_entry: &mut self.on_entry_sets.bits[range.clone()],
            gen_set: &mut self.gen_sets.bits[range.clone()],
            kill_set: &mut self.kill_sets.bits[range],
        }
    }

    fn lookup_set_for<'a>(&self, sets: &'a Bits, block_idx: usize) -> &'a [usize] {
        let offset = self.words_per_block * block_idx;
        &sets.bits[offset..(offset + self.words_per_block)]
    }
    pub fn gen_set_for(&self, block_idx: usize) -> &[usize] {
        self.lookup_set_for(&self.gen_sets, block_idx)
    }
    pub fn kill_set_for(&self, block_idx: usize) -> &[usize] {
        self.lookup_set_for(&self.kill_sets, block_idx)
    }
    pub fn on_entry_set_for(&self, block_idx: usize) -> &[usize] {
        self.lookup_set_for(&self.on_entry_sets, block_idx)
    }
    pub fn on_exit_set_for(&self, block_idx: usize) -> Vec<usize> {
        let mut set: Vec<_> = self.on_entry_set_for(block_idx).iter()
            .map(|x|*x)
            .collect();
        bitwise(&mut set[..], self.gen_set_for(block_idx), &Union);
        bitwise(&mut set[..], self.kill_set_for(block_idx), &Subtract);
        return set;
    }
}

impl<O: BitDenotation> DataflowState<O> {
    fn each_bit<F>(&self, ctxt: &O::Ctxt, words: &[usize], mut f: F)
        where F: FnMut(usize) {
        //! Helper for iterating over the bits in a bitvector.

        let bits_per_block = self.operator.bits_per_block(ctxt);
        let usize_bits: usize = mem::size_of::<usize>() * 8;
            
        for (word_index, &word) in words.iter().enumerate() {
            if word != 0 {
                let base_index = word_index * usize_bits;
                for offset in 0..usize_bits {
                    let bit = 1 << offset;
                    if (word & bit) != 0 {
                        // NB: we round up the total number of bits
                        // that we store in any given bit set so that
                        // it is an even multiple of usize::BITS. This
                        // means that there may be some stray bits at
                        // the end that do not correspond to any
                        // actual value; that's why we first check
                        // that we are in range of bits_per_block.
                        let bit_index = base_index + offset as usize;
                        if bit_index >= bits_per_block {
                            return;
                        } else {
                            f(bit_index);
                        }
                    }
                }
            }
        }
    }

    pub fn interpret_set<'c>(&self, ctxt: &'c O::Ctxt, words: &[usize]) -> Vec<&'c O::Bit> {
        let mut v = Vec::new();
        self.each_bit(ctxt, words, |i| {
            v.push(self.operator.interpret(ctxt, i));
        });
        v
    }
}

pub trait BitwiseOperator {
    /// Joins two predecessor bits together, typically either `|` or `&`
    fn join(&self, pred1: usize, pred2: usize) -> usize;
}

/// Parameterization for the precise form of data flow that is used.
pub trait DataflowOperator : BitwiseOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn initial_value() -> bool;
}

pub trait BitDenotation: DataflowOperator {
    /// Specifies what is represented by each bit in the dataflow bitvector.
    type Bit;

    /// Specifies what, if any, separate context needs to be supplied for methods below.
    type Ctxt;

    /// A name describing the dataflow analysis that this
    /// BitDenotation is supporting.  The name should be something
    /// suitable for plugging in as part of a filename e.g. avoid
    /// space-characters or other things that tend to look bad on a
    /// file system, like slashes or periods. It is also better for
    /// the name to be reasonably short, again because it will be
    /// plugged into a filename.
    fn name() -> &'static str;
    
    /// Size of each bitvector allocated for each block in the analysis.
    fn bits_per_block(&self, &Self::Ctxt) -> usize;

    /// Provides the meaning of each entry in the dataflow bitvector.
    /// (Mostly intended for use for better debug instrumentation.)
    fn interpret<'a>(&self, &'a Self::Ctxt, idx: usize) -> &'a Self::Bit;

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects that have been
    /// established *prior* to entering the start block.
    ///
    /// (For example, establishing the call arguments.)
    ///
    /// (Typically this should only modify `sets.on_entry`, since the
    /// gen and kill sets should reflect the effects of *executing*
    /// the start block itself.)
    fn start_block_effect(&self,
                          _tcx: &TyCtxt,
                          ctxt: &Self::Ctxt,
                          mir: &Mir,
                          sets: &mut BlockSets);

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating statement.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" represnting the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The statement here is `idx_stmt.1`; `idx_stmt.0` is just
    /// an identifying index: namely, the index of the statement
    /// in the basic block.
    fn statement_effect(&self,
                        _tcx: &TyCtxt,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        idx_stmt: (usize, &repr::Statement));

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating
    /// the terminator.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" represnting the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The terminator here is `idx_term.1`; `idx_term.0` is just an
    /// identifying index: namely, the number of statements in `bb`
    /// itself.
    ///
    /// The effects applied here cannot depend on which branch the
    /// terminator took.
    fn terminator_effect(&self,
                         _tcx: &TyCtxt,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         idx_term: (usize, &repr::Terminator));

    /// Mutates the block-sets according to the (flow-dependent)
    /// effect of a successful return from a Call terminator.
    ///
    /// If basic-block BB_x ends with a call-instruction that, upon
    /// successful return, flows to BB_y, then this method will be
    /// called on the exit flow-state of BB_x in order to set up the
    /// entry flow-state of BB_y.
    ///
    /// This is used, in particular, as a special case during the
    /// "propagate" loop where all of the basic blocks are repeatedly
    /// visited. Since the effects of a Call terminator are
    /// flow-dependent, the current MIR cannot encode them via just
    /// GEN and KILL sets attached to the block, and so instead we add
    /// this extra machinery to represent the flow-dependent effect.
    ///
    /// Note: as a historical artifact, this currently takes as input
    /// the *entire* packed collection of bitvectors in `in_out`.  We
    /// might want to look into narrowing that to something more
    /// specific, just to make the interface more self-documenting.
    fn propagate_call_return(&self,
                             _tcx: &TyCtxt,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             call_bb: repr::BasicBlock,
                             dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue);
}

impl<'a, 'tcx: 'a, D> DataflowAnalysis<'a, 'tcx, D>
    where D: BitDenotation, D::Ctxt: HasMoveData<'tcx>
{
    pub fn new(tcx: &'a TyCtxt<'tcx>,
               mir: &'a Mir<'tcx>,
               ctxt: D::Ctxt,
               denotation: D) -> Self {
        let bits_per_block = denotation.bits_per_block(&ctxt);
        let usize_bits = mem::size_of::<usize>() * 8;
        let words_per_block = (bits_per_block + usize_bits - 1) / usize_bits;
        let num_blocks = mir.basic_blocks.len();
        let num_words = num_blocks * words_per_block;

        let entry = if D::initial_value() { usize::MAX } else {0};

        let zeroes = Bits::new(0, num_words);
        let on_entry = Bits::new(entry, num_words);

        DataflowAnalysis { flow_state: DataflowState {
            sets: AllSets {
                bits_per_block: bits_per_block,
                words_per_block: words_per_block,
                gen_sets: zeroes.clone(),
                kill_sets: zeroes,
                on_entry_sets: on_entry,
            },
            operator: denotation,
        },
                           ctxt: ctxt,
                           mir: mir,
                           tcx: tcx,
        }
                           
    }
}

impl<'a, 'tcx: 'a, D> DataflowAnalysis<'a, 'tcx, D>
    where D: BitDenotation, D::Ctxt: HasMoveData<'tcx>
{
    /// Propagates the bits of `in_out` into all the successors of `bb`,
    /// using bitwise operator denoted by `self.operator`.
    ///
    /// For most blocks, this is entirely uniform. However, for blocks
    /// that end with a call terminator, the effect of the call on the
    /// dataflow state may depend on whether the call returned
    /// successfully or unwound.
    ///
    /// To reflect this, the `propagate_call_return` method of the
    /// `BitDenotation` mutates `in_out` when propagating `in_out` via
    /// a call terminator; such mutation is performed *last*, to
    /// ensure its side-effects do not leak elsewhere (e.g. into
    /// unwind target).
    fn propagate_bits_into_graph_successors_of(
        &mut self,
        in_out: &mut [usize],
        changed: &mut bool,
        (bb, bb_data): (repr::BasicBlock, &repr::BasicBlockData))
    {
        match bb_data.terminator().kind {
            repr::TerminatorKind::Return |
            repr::TerminatorKind::Resume => {}
            repr::TerminatorKind::Goto { ref target } |
            repr::TerminatorKind::Drop { ref target, value: _, unwind: None } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, target);
            }
            repr::TerminatorKind::Drop { ref target, value: _, unwind: Some(ref unwind) } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, target);
                self.propagate_bits_into_entry_set_for(in_out, changed, unwind);
            }
            repr::TerminatorKind::If { ref targets, .. } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, &targets.0);
                self.propagate_bits_into_entry_set_for(in_out, changed, &targets.1);
            }
            repr::TerminatorKind::Switch { ref targets, .. } |
            repr::TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets {
                    self.propagate_bits_into_entry_set_for(in_out, changed, target);
                }
            }
            repr::TerminatorKind::Call { ref cleanup, ref destination, func: _, args: _ } => {
                if let Some(ref unwind) = *cleanup {
                    self.propagate_bits_into_entry_set_for(in_out, changed, unwind);
                }
                if let Some((ref dest_lval, ref dest_bb)) = *destination {
                    // N.B.: This must be done *last*, after all other
                    // propagation, as documented in comment above.
                    self.flow_state.operator.propagate_call_return(
                        self.tcx, &self.ctxt, in_out, bb, *dest_bb, dest_lval);
                    self.propagate_bits_into_entry_set_for(in_out, changed, dest_bb);
                }
            }
        }
    }

    fn propagate_bits_into_entry_set_for(&mut self,
                                         in_out: &[usize],
                                         changed: &mut bool,
                                         bb: &repr::BasicBlock) {
        let entry_set = self.flow_state.sets.for_block(bb.index()).on_entry;
        let set_changed = bitwise(entry_set, in_out, &self.flow_state.operator);
        if set_changed {
            *changed = true;
        }
    }
}

// Dataflow analyses are built upon some interpretation of the
// bitvectors attached to each basic block, represented via a
// zero-sized structure.
//
// Note on PhantomData: Each interpretation will need to instantiate
// the `Bit` and `Ctxt` associated types, and in this case, those
// associated types need an associated lifetime `'tcx`. The
// interpretive structures are zero-sized, so they all need to carry a
// `PhantomData` representing how the structures relate to the `'tcx`
// lifetime.
//
// But, since all of the uses of `'tcx` are solely via instances of
// `Ctxt` that are passed into the `BitDenotation` methods, we can
// consistently use a `PhantomData` that is just a function over a
// `&Ctxt` (== `&MoveData<'tcx>).

/// `MaybeInitializedLvals` tracks all l-values that might be
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // maybe-init:
///                                            // {}
///     let a = S; let b = S; let c; let d;    // {a, b}
///
///     if pred {
///         drop(a);                           // {   b}
///         b = S;                             // {   b}
///
///     } else {
///         drop(b);                           // {a}
///         d = S;                             // {a,       d}
///
///     }                                      // {a, b,    d}
///
///     c = S;                                 // {a, b, c, d}
/// }
/// ```
///
/// To determine whether an l-value *must* be initialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeUninitializedLvals` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeUninitializedLvals` yields the set of
/// l-values that would require a dynamic drop-flag at that statement.
#[derive(Debug, Default)]
pub struct MaybeInitializedLvals<'a, 'tcx: 'a> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<Fn(&'a MoveData<'tcx>, &'a TyCtxt<'tcx>, &'a Mir<'tcx>)>,
}

/// `MaybeUninitializedLvals` tracks all l-values that might be
/// uninitialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // maybe-uninit:
///                                            // {a, b, c, d}
///     let a = S; let b = S; let c; let d;    // {      c, d}
///
///     if pred {
///         drop(a);                           // {a,    c, d}
///         b = S;                             // {a,    c, d}
///
///     } else {
///         drop(b);                           // {   b, c, d}
///         d = S;                             // {   b, c   }
///
///     }                                      // {a, b, c, d}
///
///     c = S;                                 // {a, b,    d}
/// }
/// ```
///
/// To determine whether an l-value *must* be uninitialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeInitializedLvals` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeInitializedLvals` yields the set of
/// l-values that would require a dynamic drop-flag at that statement.
#[derive(Debug, Default)]
pub struct MaybeUninitializedLvals<'tcx> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<for <'a> Fn(&'a MoveData<'tcx>)>,
}

/// `MovingOutStatements` tracks the statements that perform moves out
/// of particular l-values. More precisely, it tracks whether the
/// *effect* of such moves (namely, the uninitialization of the
/// l-value in question) can reach some point in the control-flow of
/// the function, or if that effect is "killed" by some intervening
/// operation reinitializing that l-value.
///
/// The resulting dataflow is a more enriched version of
/// `MaybeUninitializedLvals`. Both structures on their own only tell
/// you if an l-value *might* be uninitialized at a given point in the
/// control flow. But `MovingOutStatements` also includes the added
/// data of *which* particular statement causing the deinitialization
/// that the borrow checker's error meessage may need to report.
#[derive(Debug, Default)]
pub struct MovingOutStatements<'tcx> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<for <'a> Fn(&'a MoveData<'tcx>)>,
}

impl<'tcx> BitDenotation for MovingOutStatements<'tcx> {
    type Bit = MoveOut;
    type Ctxt = MoveData<'tcx>;
    fn name() -> &'static str { "moving_out" }
    fn bits_per_block(&self, ctxt: &MoveData<'tcx>) -> usize {
        ctxt.moves.len()
    }
    fn interpret<'c>(&self, ctxt: &'c MoveData<'tcx>, idx: usize) -> &'c Self::Bit {
        &ctxt.moves[idx]
    }
    fn start_block_effect(&self,
                          _tcx: &TyCtxt,
                          _move_data: &Self::Ctxt,
                          _mir: &Mir,
                          _sets: &mut BlockSets) {
        // no move-statements have been executed prior to function
        // execution, so this method has no effect on `_sets`.
    }
    fn statement_effect(&self,
                        _tcx: &TyCtxt,
                        move_data: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        (idx, stmt): (usize, &repr::Statement)) {
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;

        let loc = Location { block: bb, index: idx };
        debug!("stmt {:?} at loc {:?} moves out of move_indexes {:?}",
               stmt, loc, &loc_map[loc]);
        for move_index in &loc_map[loc] {
            // Every path deinitialized by a *particular move*
            // has corresponding bit, "gen'ed" (i.e. set)
            // here, in dataflow vector
            zero_to_one(&mut sets.gen_set, *move_index);
        }
        let bits_per_block = self.bits_per_block(move_data);
        match stmt.kind {
            repr::StatementKind::Assign(ref lvalue, _) => {
                // assigning into this `lvalue` kills all
                // MoveOuts from it, and *also* all MoveOuts
                // for children and associated fragment sets.
                let move_path_index = rev_lookup.find(lvalue);

                sets.kill_set.set_bit(move_path_index.idx());
                on_all_children_bits(sets.kill_set,
                                     path_map,
                                     move_paths,
                                     move_path_index,
                                     &|kill_set, mpi| {
                                         assert!(mpi.idx() < bits_per_block);
                                         kill_set.set_bit(mpi.idx());
                                     });
            }
        }
    }

    fn terminator_effect(&self,
                         tcx: &TyCtxt,
                         move_data: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         (statements_len, term): (usize, &repr::Terminator)) {
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;
        let loc = Location { block: bb, index: statements_len };
        debug!("terminator {:?} at loc {:?} moves out of move_indexes {:?}",
               term, loc, &loc_map[loc]);
        let bits_per_block = self.bits_per_block(move_data);
        for move_index in &loc_map[loc] {
            assert!(move_index.idx() < bits_per_block);
            zero_to_one(&mut sets.gen_set, *move_index);
        }
    }

    fn propagate_call_return(&self,
                             tcx: &TyCtxt,
                             move_data: &Self::Ctxt,
                             in_out: &mut [usize],
                             call_bb: repr::BasicBlock,
                             dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        let move_path_index = move_data.rev_lookup.find(dest_lval);
        let bits_per_block = self.bits_per_block(move_data);

        in_out.clear_bit(move_path_index.idx());
        on_all_children_bits(in_out,
                             &move_data.path_map,
                             &move_data.move_paths,
                             move_path_index,
                             &|in_out, mpi| {
                                 assert!(mpi.idx() < bits_per_block);
                                 in_out.clear_bit(mpi.idx());
                             });
    }
}

impl<'a, 'tcx> BitDenotation for MaybeInitializedLvals<'a, 'tcx> {
    type Bit = MovePath<'tcx>;
    type Ctxt = (&'a TyCtxt<'tcx>, &'a Mir<'tcx>, MoveData<'tcx>);
    fn name() -> &'static str { "maybe_init" }
    fn bits_per_block(&self, ctxt: &Self::Ctxt) -> usize {
        ctxt.2.move_paths.len()
    }
    fn interpret<'c>(&self, ctxt: &'c Self::Ctxt, idx: usize) -> &'c Self::Bit {
        &ctxt.2.move_paths[MovePathIndex::new(idx)]
    }

    // sets on_entry bits for Arg lvalues
    fn start_block_effect(&self,
                          tcx: &TyCtxt,
                          ctxt: &Self::Ctxt,
                          mir: &Mir,
                          sets: &mut BlockSets) {
        let move_data = &ctxt.2;
        let bits_per_block = self.bits_per_block(ctxt);

        on_all_children_of_arg_lvalues(
            move_data, mir, sets, &|on_entry, moi| {
                let mpi = moi.move_path_index(move_data);
                assert!(mpi.idx() < bits_per_block);
                on_entry.set_bit(mpi.idx());
            });

        // let move_paths = &move_data.move_paths;
        // let path_map = &move_data.path_map;
        // let rev_lookup = &move_data.rev_lookup;
        // 
        // for i in 0..(mir.arg_decls.len() as u32) {
        //     let lvalue = repr::Lvalue::Arg(i);
        //     let move_path_index = rev_lookup.find(&lvalue);
        //     on_all_children_bits(sets.on_entry,
        //                          path_map,
        //                          move_paths,
        //                          move_path_index,
        //                          &|on_entry, moi| {
        //                              let mpi = moi.move_path_index(move_data);
        //                              assert!(mpi.idx() < bits_per_block);
        //                              on_entry.set_bit(mpi.idx());
        //                          });
        // }
    }

    // gens bits for lvalues initialized by statement
    // kills bits for lvalues moved-out by statement
    fn statement_effect(&self,
                        tcx: &TyCtxt,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        (idx, stmt): (usize, &repr::Statement)) {
        let tcx = ctxt.0;
        let mir = ctxt.1;
        let move_data = &ctxt.2;
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;

        let loc = Location { block: bb, index: idx };
        let move_outs = &loc_map[loc];

        // first, setup kill_set: kill bits for l-values moved out by
        // stmt (`path` intermediate vec is unecessary, but arguably
        // result debug! printout better reflects method's effects).
        let paths = move_outs_paths(move_data, &move_outs[..]);
        debug!("stmt {:?} at loc {:?} moves out of paths {:?}",
               stmt, loc, paths);
        let bits_per_block = self.bits_per_block(ctxt);
        for &move_path_index in &paths {

            // first, check if the lvalue's type implements Copy, in
            // which case uses do not kill it.
            if let MovePathContent::Lvalue(ref lvalue) = move_data.move_paths[move_path_index].content {
                let ty = mir.lvalue_ty(tcx, lvalue).to_ty(tcx);
                let empty_param_env = tcx.empty_parameter_environment();
                if !ty.moves_by_default(&empty_param_env, DUMMY_SP) {
                    continue;
                }
            }

            // (don't use zero_to_one since bit may already be set to 1.)
            sets.kill_set.set_bit(move_path_index.idx());
            on_all_children_bits(sets.kill_set,
                                 path_map,
                                 move_paths,
                                 move_path_index,
                                 &|kill_set, moi| {
                                     let mpi = moi.move_path_index(move_data);
                                     if mpi.idx() >= bits_per_block {
                                         debug!("child bit {} is out of range {}",
                                                mpi.idx(), bits_per_block);
                                     }
                                     assert!(mpi.idx() < bits_per_block);
                                     kill_set.set_bit(mpi.idx());
                                 });
        }

        // second, the gen_set: initialized l-values are generated,
        // and removed from kill_set (because dataflow will first
        // apply gen_set effects, followed by the kill_set effects).
        match stmt.kind {
            repr::StatementKind::Assign(ref lvalue, _) => {
                // assigning into `lvalue` gens a bit for it and *also*
                // all lvalues for children and associated fragment sets.
                //
                // also clear the corresponding bit in the kill set, if any
                let move_path_index = rev_lookup.find(lvalue);

                sets.gen_set.set_bit(move_path_index.idx());
                on_all_children_bits(sets.gen_set,
                                     path_map,
                                     move_paths,
                                     move_path_index,
                                     &|gen_set, moi| {
                                         let mpi = moi.move_path_index(move_data);
                                         assert!(mpi.idx() < bits_per_block);
                                         gen_set.set_bit(mpi.idx());
                                     });

                sets.kill_set.clear_bit(move_path_index.idx());
                on_all_children_bits(sets.kill_set,
                                     path_map,
                                     move_paths,
                                     move_path_index,
                                     &|kill_set, moi| {
                                         let mpi = moi.move_path_index(move_data);
                                         assert!(mpi.idx() < bits_per_block);
                                         kill_set.clear_bit(mpi.idx());
                                     });
            }
        }
    }

    fn terminator_effect(&self,
                         tcx: &TyCtxt,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         (statements_len, term): (usize, &repr::Terminator)) {
        let move_data = &ctxt.2;
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;
        let loc = Location { block: bb, index: statements_len };
        let paths = move_outs_paths(move_data, &loc_map[loc]);
        debug!("terminator {:?} at loc {:?} moves out of paths {:?}",
               term, loc, &paths);
        let bits_per_block = self.bits_per_block(ctxt);
        for &move_path_index in &paths {
            // (don't use zero_to_one since bit may already be set to 1.)
            sets.kill_set.set_bit(move_path_index.idx());
            on_all_children_bits(sets.kill_set,
                                 path_map,
                                 move_paths,
                                 move_path_index,
                                 &|kill_set, moi| {
                                     let mpi = moi.move_path_index(move_data);
                                     assert!(mpi.idx() < bits_per_block);
                                     kill_set.set_bit(mpi.idx());
                                 });
        }
    }

    fn propagate_call_return(&self,
                             tcx: &TyCtxt,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             call_bb: repr::BasicBlock,
                             dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 1 (initialized).
        let move_data = &ctxt.2;
        let move_path_index = move_data.rev_lookup.find(dest_lval);
        let bits_per_block = self.bits_per_block(ctxt);
        in_out.set_bit(move_path_index.idx());
        on_all_children_bits(in_out,
                             &move_data.path_map,
                             &move_data.move_paths,
                             move_path_index,
                             &|in_out, moi| {
                                 let mpi = moi.move_path_index(move_data);
                                 assert!(mpi.idx() < bits_per_block);
                                 in_out.set_bit(mpi.idx());
                             });
    }
}

impl<'tcx> BitDenotation for MaybeUninitializedLvals<'tcx> {
    type Bit = MovePath<'tcx>;
    type Ctxt = MoveData<'tcx>;
    fn name() -> &'static str { "maybe_uninit" }
    fn bits_per_block(&self, ctxt: &Self::Ctxt) -> usize {
        ctxt.move_paths.len()
    }
    fn interpret<'c>(&self, ctxt: &'c Self::Ctxt, idx: usize) -> &'c Self::Bit {
        &ctxt.move_paths[MovePathIndex::new(idx)]
    }

    // sets on_entry bits for Arg lvalues
    fn start_block_effect(&self,
                          tcx: &TyCtxt,
                          move_data: &Self::Ctxt,
                          mir: &Mir,
                          sets: &mut BlockSets) {
        let bits_per_block = self.bits_per_block(move_data);

        on_all_children_of_arg_lvalues(
            move_data, mir, sets, &|on_entry, moi| {
                let mpi = moi.move_path_index(move_data);
                assert!(mpi.idx() < bits_per_block);
                on_entry.clear_bit(mpi.idx());
            });
    }

    // gens bits for lvalues moved-out by statement
    // kills bits for lvalues initialized by statement
    fn statement_effect(&self,
                        tcx: &TyCtxt,
                        move_data: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        (idx, stmt): (usize, &repr::Statement)) {
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;

        let loc = Location { block: bb, index: idx };
        let move_outs = &loc_map[loc];

        // first, setup gen_set: gen bits for l-values moved out by
        // stmt (`path` intermediate vec is unecessary, but arguably
        // result debug! printout better reflects method's effects).
        let paths = move_outs_paths(move_data, &move_outs[..]);
        debug!("stmt {:?} at loc {:?} moves out of paths {:?}",
               stmt, loc, paths);
        let bits_per_block = self.bits_per_block(move_data);
        for &move_path_index in &paths {
            // (don't use zero_to_one since bit may already be set to 1.)
            sets.gen_set.set_bit(move_path_index.idx());
            on_all_children_bits(sets.gen_set,
                                 path_map,
                                 move_paths,
                                 move_path_index,
                                 &|gen_set, moi| {
                                     let mpi = moi.move_path_index(move_data);
                                     assert!(mpi.idx() < bits_per_block);
                                     gen_set.set_bit(mpi.idx());
                                 });
        }

        // second, the kill_set: kill bits for initialized l-value, if any
        match stmt.kind {
            repr::StatementKind::Assign(ref lvalue, _) => {
                // assigning into `lvalue` gens a bit for it and *also*
                // all lvalues for children and associated fragment sets.
                //
                // also clear the corresponding bit in the kill set, if any
                let move_path_index = rev_lookup.find(lvalue);

                sets.kill_set.set_bit(move_path_index.idx());
                on_all_children_bits(sets.kill_set,
                                     path_map,
                                     move_paths,
                                     move_path_index,
                                     &|kill_set, moi| {
                                         let mpi = moi.move_path_index(move_data);
                                         assert!(mpi.idx() < bits_per_block);
                                         kill_set.set_bit(mpi.idx());
                                     });
            }
        }
    }

    fn terminator_effect(&self,
                         tcx: &TyCtxt,
                         move_data: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         (statements_len, term): (usize, &repr::Terminator)) {
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;
        let loc = Location { block: bb, index: statements_len };
        let paths = move_outs_paths(move_data, &loc_map[loc]);
        debug!("terminator {:?} at loc {:?} moves out of paths {:?}",
               term, loc, paths);
        let bits_per_block = self.bits_per_block(move_data);
        for &move_path_index in &paths[..] {
            // (don't use zero_to_one since bit may already be set to 1.)
            sets.kill_set.set_bit(move_path_index.idx());
            on_all_children_bits(sets.kill_set,
                                 path_map,
                                 move_paths,
                                 move_path_index,
                                 &|kill_set, moi| {
                                     let mpi = moi.move_path_index(move_data);
                                     assert!(mpi.idx() < bits_per_block);
                                     kill_set.set_bit(mpi.idx());
                                 });
        }
    }


    fn propagate_call_return(&self,
                             tcx: &TyCtxt,
                             move_data: &Self::Ctxt,
                             in_out: &mut [usize],
                             call_bb: repr::BasicBlock,
                             dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        // when a call returns successfully, that means we need to
        // clear the bits for that (definitely initialized) dest_lval.
        let move_path_index = move_data.rev_lookup.find(dest_lval);
        let bits_per_block = self.bits_per_block(move_data);
        in_out.clear_bit(move_path_index.idx());
        on_all_children_bits(in_out,
                             &move_data.path_map,
                             &move_data.move_paths,
                             move_path_index,
                             &|in_out, moi| {
                                 let mpi = moi.move_path_index(move_data);
                                 assert!(mpi.idx() < bits_per_block);
                                 in_out.clear_bit(mpi.idx());
                             });
    }
}

fn zero_to_one(bitvec: &mut [usize], move_index: MoveOutIndex) {
    let retval = bitvec.set_bit(move_index.idx());
    assert!(retval);
}

fn move_outs_paths(move_data: &MoveData,
                   move_outs: &[MoveOutIndex]) -> Vec<MovePathIndex> {
    move_outs.iter()
        .map(|mi| move_data.moves[mi.idx()].path)
        .collect()
}

fn on_all_children_of_arg_lvalues<Each>(move_data: &MoveData,
                                        mir: &Mir,
                                        sets: &mut BlockSets,
                                        each_child: &Each)
    where Each: Fn(&mut [usize], MoveOutIndex)
{
    let move_paths = &move_data.move_paths;
    let path_map = &move_data.path_map;
    let rev_lookup = &move_data.rev_lookup;

    for i in 0..(mir.arg_decls.len() as u32) {
        let lvalue = repr::Lvalue::Arg(i);
        let move_path_index = rev_lookup.find(&lvalue);
        on_all_children_bits(sets.on_entry,
                             path_map,
                             move_paths,
                             move_path_index,
                             each_child);
    }
}

impl<'tcx> BitwiseOperator for MovingOutStatements<'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // moves from both preds are in scope
    }
}

impl<'a, 'tcx> BitwiseOperator for MaybeInitializedLvals<'a, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // "maybe" means we union effects of both preds
    }
}

impl<'tcx> BitwiseOperator for MaybeUninitializedLvals<'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // "maybe" means we union effects of both preds
    }
}

impl<'tcx> DataflowOperator for MovingOutStatements<'tcx> {
    #[inline]
    fn initial_value() -> bool {
        false // no loans in scope by default
    }
}

impl<'a, 'tcx> DataflowOperator for MaybeInitializedLvals<'a, 'tcx> {
    #[inline]
    fn initial_value() -> bool {
        false // lvalues start uninitialized
    }
}

impl<'tcx> DataflowOperator for MaybeUninitializedLvals<'tcx> {
    #[inline]
    fn initial_value() -> bool {
        false // lvalues start uninitialized
    }
}

#[inline]
fn bitwise<Op:BitwiseOperator>(out_vec: &mut [usize],
                               in_vec: &[usize],
                               op: &Op) -> bool {
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elt, in_elt) in out_vec.iter_mut().zip(in_vec) {
        let old_val = *out_elt;
        let new_val = op.join(old_val, *in_elt);
        *out_elt = new_val;
        changed |= old_val != new_val;
    }
    changed
}

struct Union;
impl BitwiseOperator for Union {
    fn join(&self, a: usize, b: usize) -> usize { a | b }
}
struct Subtract;
impl BitwiseOperator for Subtract {
    fn join(&self, a: usize, b: usize) -> usize { a & !b }
}
