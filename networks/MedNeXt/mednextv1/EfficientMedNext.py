import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from networks.MedNeXt.mednextv1.efficient_mednext_blocks import *

class EfficientMedNeXt(nn.Module):

    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        kernel_sizes = [1,3,5],                      # Ofcourse can test kernel_size
        strides=[1,1,1],
        enc_kernel_sizes: int = [1,3,5],
        dec_kernel_sizes: int = [1,3,5],
        uniform_dec_channels = None,
        deep_supervision: bool = True,      #False       # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False,
        mode = 'train'
    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']
        print(kernel_sizes)
        if kernel_sizes is not None:
            enc_kernel_sizes = kernel_sizes
            dec_kernel_sizes = kernel_sizes
        up_down_strides = [s*2 for s in strides]
        
        print(strides, up_down_strides)
        
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        
        num_channels =  [n_channels, n_channels*2, n_channels*4, n_channels*8, n_channels*16]
        
        if uniform_dec_channels == None:
            dec_num_channels =  [nc for nc in num_channels] 
        else:
            dec_num_channels =  [uniform_dec_channels for i in range(len(num_channels))] 
        print(dec_num_channels)  
        self.stem = conv(in_channels, num_channels[0], kernel_size=1)
        
        self.enc_block_0 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=num_channels[0],
                out_channels=num_channels[0],
                kernel_sizes=enc_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                ) 
            for i in range(block_counts[0])]
        ) 

        self.down_0 = EfficientMedNeXtDownBlock(
            in_channels=num_channels[0],
            out_channels=num_channels[1],
            kernel_sizes=enc_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )
    
        self.enc_block_1 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=num_channels[1],
                out_channels=num_channels[1],
                kernel_sizes=enc_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = EfficientMedNeXtDownBlock(
            in_channels=num_channels[1],
            out_channels=num_channels[2],
            kernel_sizes=enc_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=num_channels[2],
                out_channels=num_channels[2],
                kernel_sizes=enc_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = EfficientMedNeXtDownBlock(
            in_channels=num_channels[2],
            out_channels=num_channels[3],
            kernel_sizes=enc_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        self.enc_block_3 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=num_channels[3],
                out_channels=num_channels[3],
                kernel_sizes=enc_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )            
            for i in range(block_counts[3])]
        )
        
        self.down_3 = EfficientMedNeXtDownBlock(
            in_channels=num_channels[3],
            out_channels=num_channels[4],
            kernel_sizes=enc_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.crrb4 = EfficientMedNeXtBlock(
                in_channels=num_channels[4],
                out_channels=dec_num_channels[4],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
        
        self.bottleneck = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=dec_num_channels[4],
                out_channels=dec_num_channels[4],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[4])]
        )

        self.crrb3 = EfficientMedNeXtBlock(
                in_channels=num_channels[3],
                out_channels=dec_num_channels[3],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
        self.crrb2 = EfficientMedNeXtBlock(
                in_channels=num_channels[2],
                out_channels=dec_num_channels[2],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
        
        self.crrb1 = EfficientMedNeXtBlock(
                in_channels=num_channels[1],
                out_channels=dec_num_channels[1],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
        if mode == 'train':
            self.crrb0 = EfficientMedNeXtBlock(
                in_channels=num_channels[0],
                out_channels=dec_num_channels[0],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            
        self.up_3 = EfficientMedNeXtUpBlock(
            in_channels=dec_num_channels[4],
            out_channels=dec_num_channels[3],
            kernel_sizes=dec_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=dec_num_channels[3],
                out_channels=dec_num_channels[3],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[5])]
        )

        self.up_2 = EfficientMedNeXtUpBlock(
            in_channels=dec_num_channels[3],
            out_channels=dec_num_channels[2],
            kernel_sizes=dec_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=dec_num_channels[2],
                out_channels=dec_num_channels[2],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = EfficientMedNeXtUpBlock(
            in_channels=dec_num_channels[2],
            out_channels=dec_num_channels[1],
            kernel_sizes=dec_kernel_sizes,
            strides = up_down_strides,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            EfficientMedNeXtBlock(
                in_channels=dec_num_channels[1],
                out_channels=dec_num_channels[1],
                kernel_sizes=dec_kernel_sizes,
                strides = strides,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )
        if mode == 'train':
            self.up_0 = EfficientMedNeXtUpBlock(
                in_channels=dec_num_channels[1],
                out_channels=dec_num_channels[0],
                kernel_sizes=dec_kernel_sizes,
                strides = up_down_strides,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
        
            self.dec_block_0 = nn.Sequential(*[
                EfficientMedNeXtBlock(
                    in_channels=dec_num_channels[0],
                    out_channels=dec_num_channels[0],
                    kernel_sizes=dec_kernel_sizes,
                    strides = strides,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn
                )
                for i in range(block_counts[8])]
            )
            self.out_0 = OutBlock(in_channels=dec_num_channels[0], n_classes=n_classes, dim=dim)
        
        self.out_1 = OutBlock(in_channels=dec_num_channels[1], n_classes=n_classes, dim=dim)#, stride=2)
            
        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)  

        if deep_supervision and mode == 'train':
            self.out_2 = OutBlock(in_channels=dec_num_channels[2], n_classes=n_classes, dim=dim)#, stride=4)
            self.out_3 = OutBlock(in_channels=dec_num_channels[3], n_classes=n_classes, dim=dim)#, stride=8)
            self.out_4 = OutBlock(in_channels=dec_num_channels[4], n_classes=n_classes, dim=dim)#, stride=16)

        self.block_counts = block_counts


    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x


    def forward(self, x, mode='test'):
        
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            if mode != 'test':
                x_res_0 = checkpoint.checkpoint(self.crrb0, x_res_0, self.dummy_tensor)
            x_res_1 = checkpoint.checkpoint(self.crrb1, x_res_1, self.dummy_tensor)
            x_res_2 = checkpoint.checkpoint(self.crrb2, x_res_2, self.dummy_tensor)
            x_res_3 = checkpoint.checkpoint(self.crrb3, x_res_3, self.dummy_tensor)
            x = checkpoint.checkpoint(self.crrb4, x, self.dummy_tensor)
            
            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3 
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2 
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1 
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            del x_res_1, x_up_1
            if self.do_ds or mode =='test':
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            #Newly added start
            if mode == 'test':
                #print('returning early')
                return [x_ds_1]
            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0 
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)
            #newly added end
            
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            if mode != 'test':
                x_res_0 = self.crrb0(x_res_0)
            x_res_1 = self.crrb1(x_res_1)
            x_res_2 = self.crrb2(x_res_2)
            x_res_3 = self.crrb3(x_res_3)
            x = self.crrb4(x)
            
            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3 
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2 
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1 
            x = self.dec_block_1(dec_x)
            del x_res_1, x_up_1
            if self.do_ds or mode =='test':
                x_ds_1 = self.out_1(x)
            
            #newly added start
            if mode == 'test':
                return [x_ds_1]
            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0 
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x
            x = self.out_0(x)
            #newly added end
            
        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            #print('returning final')
            return x


if __name__ == "__main__":

    network = EfficientMedNeXt(
            in_channels = 1, 
            n_channels = 32,
            n_classes = 13,
            kernel_sizes=[1,3,5],                     # Can test kernel_size
            strides=[1,1,1],
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = None,
            dim = '2d',
            grn=True
            
        ).cuda()
    

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1,1,64,64,64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())
    
    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)
