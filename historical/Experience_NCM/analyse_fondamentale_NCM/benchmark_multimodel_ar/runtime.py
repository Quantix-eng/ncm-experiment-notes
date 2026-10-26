"""Layer-continuation adapter, keeping only recipient's earlier-token context."""
from support import *


class Engine:
    def __init__(self, net):
        self.net = net
        self.inputs, self.outputs, self.attention, self.mlp = {}, {}, {}, {}
        self.handles = []
        for i, block in enumerate(net.transformer.h):
            def before(module, args, i=i):
                self.inputs[i] = args[0].detach().clone()
            def after(module, args, output, i=i):
                self.outputs[i] = output[0].detach().clone()
            def attn(module, args, output, i=i):
                self.attention[i] = output[0][0, -1].detach().clone()
            def mlp(module, args, output, i=i):
                self.mlp[i] = output[0, -1].detach().clone()
            self.handles.extend([block.register_forward_pre_hook(before), block.register_forward_hook(after),
                                 block.attn.register_forward_hook(attn), block.mlp.register_forward_hook(mlp)])

    def decode(self, context, z):
        full = context['full'][-1].clone()
        full[0, -1] = z
        return self.net.lm_head(self.net.transformer.ln_f(full))[0, -1]

    @staticmethod
    def frozen(z, tail):
        result = z.clone()
        for i, contribution in enumerate(tail):
            result = contribution+result if i % 2 == 0 else result+contribution
        return result

    def native(self, tokens):
        expected = self.net(torch.tensor([tokens]), use_cache=False).logits[0, -1].detach().clone()
        full = torch.stack([self.inputs[0]]+[self.outputs[i] for i in range(12)])
        # Clone all saved values before subsequent decoder/block calls overwrite hooks.
        context = dict(full=full, tokens=tokens,
                       tail=torch.stack([value for i in range(6, 12)
                                         for value in (self.attention[i], self.mlp[i])]))
        logits = torch.stack([self.decode(context, state[0, -1]) for state in full])
        assert torch.equal(logits[-1], expected), 'Official/native decoder mismatch'
        z = full[6, 0, -1]
        assert torch.equal(self.frozen(z, context['tail']), full[-1, 0, -1])
        regenerated = self.continue_from(context, z)
        assert torch.equal(regenerated['tail'], context['tail']), 'Native tail mismatch'
        assert torch.equal(regenerated['logits'], expected), 'Sham mismatch'
        return context, logits

    def continue_from(self, context, z, explicit=False):
        result = z.clone()
        tail = []
        for i in range(6, 12):
            full = context['full'][i].clone()
            full[0, -1] = result
            block = self.net.transformer.h[i]
            if explicit:
                attention = block.attn(block.ln_1(full), use_cache=False)[0]
                mid = attention+full
                feedforward = block.mlp(block.ln_2(mid))
                out = mid+feedforward
                a, m = attention[0, -1].clone(), feedforward[0, -1].clone()
            else:
                out = block(full, use_cache=False)[0]
                a, m = self.attention[i].clone(), self.mlp[i].clone()
            assert torch.equal(out[0, :-1], context['full'][i+1, 0, :-1]), 'Earlier-token context changed'
            tail.extend((a, m))
            result = out[0, -1].clone()
        contributions = torch.stack(tail)
        assert torch.equal(self.frozen(z, contributions), result)
        return dict(z=result, tail=contributions, logits=self.decode(context, result))

    def close(self):
        for handle in self.handles:
            handle.remove()
