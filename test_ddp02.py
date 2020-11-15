# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 test_ddp02.py
import torch
import argparse
import torch.distributed as dist

weight = torch.zeros(1)
gather_list = [torch.ones_like(weight) for i in range(2)]

# gloo
def run(rank):
    global weight

    if rank == 0:
        dist.gather(weight, gather_list, 0)
        print(gather_list)
    else:
        weight = torch.randn(1)
        dist.gather(weight)
        print('rank {} is sending data {} to rank 0'
              .format(rank, weight))


# nccl
""" All-Reduce example."""
def run1(rank):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


'''All-gather example'''
def run2(rank):
    group = dist.new_group([0, 1])
    weight = torch.randn(1).cuda()
    gather_list = [torch.ones_like(weight) for i in range(2)]
    dist.all_gather(gather_list, weight, group=group)
    dist.barrier()
    print(gather_list)


if __name__ == '__main__':
    print('process starts')
    # dist.init_process_group(backend='gloo')
    dist.init_process_group(backend='nccl')
    run2(dist.get_rank())
