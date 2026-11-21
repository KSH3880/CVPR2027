import os

import torch

CARRY_MODE = os.environ.get('SINGLE_CARRY', 'table1')
if CARRY_MODE not in ('table1', 'original9'):
    raise ValueError('Expected SINGLE_CARRY=table1/original9')
NINE_SIZES = ((.22, .22, .22), (.27, .27, .27), (.32, .32, .32),
              (.37, .37, .37), (.42, .42, .42), (.47, .47, .47),
              (.52, .52, .52), (.57, .57, .57), (.30, .30, .40))


def apply_original9(sizes, masses, kinds, primary, seed):
    generator = torch.Generator(device=sizes.device).manual_seed(27000+seed)
    pool = sizes.new_tensor(NINE_SIZES)
    for kind in (3, 4):
        rows = (kinds == kind).nonzero().flatten()
        count = len(rows)
        if count >= len(pool):
            ids = torch.cat((torch.arange(len(pool), device=sizes.device),
                torch.multinomial(torch.ones(len(pool), device=sizes.device), count-len(pool),
                                  replacement=True, generator=generator)))
            ids = ids[torch.randperm(count, device=sizes.device, generator=generator)]
        else:
            ids = torch.multinomial(torch.ones(len(pool), device=sizes.device), count,
                                    replacement=True, generator=generator)
        slots = primary[rows]
        density = masses[rows, slots]/sizes[rows, slots].prod(-1)
        sizes[rows, slots] = pool[ids]
        masses[rows, slots] = density*pool[ids].prod(-1)
