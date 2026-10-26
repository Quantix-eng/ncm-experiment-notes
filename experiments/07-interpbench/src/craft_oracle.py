"""Official Tracr-to-Craft compiler, stopping before optional JAX assembly.

RASP program functions are extracted unchanged from pinned official case sources.
No diagnostic outputs, learned-model outputs, or mechanism labels enter this oracle.
"""
from pathlib import Path
import ast
import copy
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources/tracr_checkout'))
sys.path.insert(0,str(HERE/'sources/github'))
from tracr.rasp import rasp
from tracr.compiler import basis_inference,craft_graph_to_model,expr_to_craft_graph,nodes,rasp_to_graph,validating
from tracr.craft import bases,transformers

META={int(x['case_id']):x for x in json.loads((HERE/'sources/benchmark_metadata.json').read_text())['cases'] if x['case_id'].isdigit()}


def official_program(case):
    source=HERE/f'sources/github/circuits_benchmark/benchmark/cases/case_{case}.py'
    tree=ast.parse(source.read_text());kept=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module in [
                'circuits_benchmark.benchmark.tracr_benchmark_case',
                'circuits_benchmark.benchmark']:
            continue
        if isinstance(node,ast.ClassDef):
            # Keep the published get_program body and all published helper functions.
            # Remove the superclass wrapper to avoid importing training dependencies.
            kept.extend(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='get_program')
        elif isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):
            kept.append(node)
        else:
            raise ValueError(f'Unexpected top-level statement in reviewed source: {ast.dump(node)}')
    module=ast.fix_missing_locations(ast.Module(body=kept,type_ignores=[]))
    namespace={'__name__':f'official_interpbench_case_{case}'}
    exec(compile(module,str(source),'exec'),namespace)
    return namespace['get_program'](None)


class PatchedRASP(rasp.DefaultRASPEvaluator):
    def __init__(self,patches):
        super().__init__();self.patches=patches
    def evaluate(self,expr,xs):
        if expr.label in self.patches:return self.patches[expr.label]
        return super().evaluate(expr,xs)


class CraftOracle:
    def __init__(self,case):
        self.case=case;self.meta=META[case];self.program=official_program(case)
        unsupported=validating.static_validate(self.program)
        if unsupported:raise ValueError(unsupported)
        extracted=rasp_to_graph.extract_rasp_graph(self.program)
        self.graph,self.sources,self.sink=extracted.graph,extracted.sources,extracted.sink
        basis_inference.infer_bases(self.graph,self.sink,set(self.meta['vocab']),self.meta['max_seq_len']-1)
        # Exact calls from the official compile_rasp_to_model, before JAX assembly.
        expr_to_craft_graph.add_craft_components_to_rasp_graph(
            self.graph,bos_dir=bases.BasisDirection(rasp.tokens.label,'BOS'),mlp_exactness=100)
        self.craft=craft_graph_to_model.craft_graph_to_model(self.graph,self.sources)
        self.space=self.craft.residual_space
        self.output=bases.VectorSpaceWithBasis(self.sink[nodes.OUTPUT_BASIS])
        self.blocks=[];candidate=iter((l,t) for l in range(len(self.craft.blocks)+1) for t in ['attn','mlp'])
        for block in self.craft.blocks:
            kind='mlp' if isinstance(block,transformers.MLP) else 'attn'
            where=next(candidate)
            while where[1]!=kind:where=next(candidate)
            layer=where[0]
            if kind=='attn':
                heads=list(block.as_multi().heads())
                for head in heads:
                    # The final official assembler applies the declared causal mask.
                    head.causal=self.meta['transformer_cfg']['attention_dir']=='causal'
                self.blocks.append((layer,kind,heads))
            else:self.blocks.append((layer,kind,block))
        self.labels={node[nodes.EXPR].label:node[nodes.EXPR] for _,node in self.graph.nodes.items()}

    def schema(self):
        result=[]
        for layer,kind,obj in self.blocks:
            parts=enumerate(obj) if kind=='attn' else [(None,obj)]
            for head,block in parts:
                out=block.w_ov.output_space if kind=='attn' else block.snd.output_space
                labels=sorted({d.name for d in out.basis})
                result.append({'layer':layer,'kind':kind,'head':head,
                               'output_labels':labels,'rasp_labels_found':[x for x in labels if x in self.labels],
                               'all_labels_found':all(x in self.labels for x in labels)})
        return result

    def embed(self,tokens):
        seq=['BOS']+list(tokens)
        matrix=np.zeros((len(seq),self.space.num_dims),dtype=np.float64)
        columns={d:i for i,d in enumerate(self.space.basis)}
        for pos,token in enumerate(seq):
            for direction in [bases.BasisDirection(rasp.tokens.label,token),bases.BasisDirection('one')]:
                if direction in columns:matrix[pos,columns[direction]]=1.
            direction=bases.BasisDirection(rasp.indices.label,pos-1)
            if pos and direction in columns:matrix[pos,columns[direction]]=1.
        return bases.VectorInBasis(self.space.basis,matrix)

    def native(self,tokens,patch=None):
        x=self.embed(tokens);cache={}
        for layer,kind,obj in self.blocks:
            if kind=='attn':
                contributions=[]
                for h,head in enumerate(obj):
                    key=(layer,kind,h)
                    out=head.apply(x.project(head.residual_space)).project(self.space)
                    if patch and key==patch[0]:out=patch[1]
                    cache[key]=out;contributions.append(out)
                if contributions:x=x+bases.VectorInBasis.sum(contributions)
            else:
                key=(layer,kind,None);out=obj.apply(x.project(obj.residual_space)).project(self.space)
                if patch and key==patch[0]:out=patch[1]
                cache[key]=out;x=x+out
        return x.project(self.output).magnitudes,cache

    def semantic(self,tokens,variable=None,donor_tokens=None):
        if variable is None:return list(self.program(tokens))
        entry=next((r for r in self.schema() if (r['layer'],r['kind'],r['head'])==variable),None)
        if entry is None:return list(self.program(tokens))  # Unaligned low-level node.
        if not entry['all_labels_found']:raise ValueError('No semantic mask for auxiliary Craft variables')
        if len(tokens)!=len(donor_tokens):raise ValueError('Native full-variable swap requires equal lengths')
        patches={label:list(rasp.evaluate(self.labels[label],donor_tokens)) for label in entry['output_labels']}
        return list(PatchedRASP(patches).evaluate(self.program,tokens))

    def target_ids(self,values):
        mapping={d.value:i for i,d in enumerate(self.output.basis)}
        return [None if v is None else mapping[v] for v in values]


def check_native(oracle,tokens):
    scores,_=oracle.native(tokens);semantic=oracle.semantic(tokens);ids=oracle.target_ids(semantic)
    mask=np.array([v is not None for v in ids]);target=np.array([0 if v is None else v for v in ids])
    return bool(np.array_equal(scores[1:].argmax(-1)[mask],target[mask])),semantic
