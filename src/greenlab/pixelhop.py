import torch
import torch.nn as nn
from .conv2d_pca import Conv2d_PCA


class PixelHopPlusPlus(nn.Module):
    def __init__(
        self,
        num_channels=[
            4,  # Hop 1
            [4, 4, 4, 4],  # Hop 2
            [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],  # Hop 3
            [
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
            ],  # Hop 4
        ],  # define the number of channels for each hop
    ):
        """
        Hop1:
        DC AC1 AC2 AC3

        Hop2:
        DC -> DC-DC DC-AC1 DC-AC2 DC-AC3
        AC1 -> AC1-DC AC1-AC1 AC1-AC2 AC1-AC3
        AC2 -> AC2-DC AC2-AC1 AC2-AC2 AC2-AC3
        AC3 -> AC3-DC AC3-AC1 AC3-AC2 AC3-AC3

        Hop3:
        DC-DC -> DC-DC-DC DC-DC-AC1 DC-DC-AC2 DC-DC-AC3
        DC-AC1 -> DC-AC1-DC DC-AC1-AC1 DC-AC1-AC2 DC-AC1-AC3
        DC-AC2 -> DC-AC2-DC DC-AC2-AC1 DC-AC2-AC2 DC-AC2-AC3
        DC-AC3 -> DC-AC3-DC DC-AC3-AC1 DC-AC3-AC2 DC-AC3-AC3

        AC1-DC -> AC1-DC-DC AC1-DC-AC1 AC1-DC-AC2 AC1-DC-AC3
        AC1-AC1 -> AC1-AC1-DC AC1-AC1-AC1 AC1-AC1-AC2 AC1-AC1-AC3
        AC1-AC2 -> AC1-AC2-DC AC1-AC2-AC1 AC1-AC2-AC2 AC1-AC2-AC3
        AC1-AC3 -> AC1-AC3-DC AC1-AC3-AC1 AC1-AC3-AC2 AC1-AC3-AC3
        ...

        Hop4:
        DC-DC-DC -> DC-DC-DC-DC DC-DC-DC-AC1 DC-DC-DC-AC2 DC-DC-DC-AC3

        """
        super(PixelHopPlusPlus, self).__init__()
        self.conv1 = Conv2d_PCA(
            in_channels=3,
            out_channels=num_channels[0],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
            mode="saab",
            name="hop1",
        )

        self.cw_conv2 = []
        for hop1_index, output_channels in enumerate(num_channels[1]):
            self.cw_conv2.append(
                Conv2d_PCA(
                    in_channels=1,
                    out_channels=output_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                    mode="saab",
                    name="hop2_hop1_ch" + str(hop1_index),
                )
            )
        self.cw_conv2 = nn.ModuleList(self.cw_conv2)

        self.cw_conv3 = []
        for hop1_index, hop2_channels in enumerate(num_channels[2]):
            self.cw_conv3.append([])
            for hop2_index, output_channels in enumerate(hop2_channels):
                self.cw_conv3[hop1_index].append(
                    Conv2d_PCA(
                        in_channels=1,
                        out_channels=output_channels,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        bias=False,
                        mode="saab",
                        name="hop3_hop1_ch"
                        + str(hop1_index)
                        + "_hop2_ch"
                        + str(hop2_index),
                    )
                )
            self.cw_conv3[hop1_index] = nn.ModuleList(self.cw_conv3[hop1_index])
        self.cw_conv3 = nn.ModuleList(self.cw_conv3)

        self.cw_conv4 = []
        for hop1_index, hop2_channels in enumerate(num_channels[3]):
            self.cw_conv4.append([])
            for hop2_index, hop3_channels in enumerate(hop2_channels):
                self.cw_conv4[hop1_index].append([])
                for hop3_index, output_channels in enumerate(hop3_channels):
                    self.cw_conv4[hop1_index][hop2_index].append(
                        Conv2d_PCA(
                            in_channels=1,
                            out_channels=output_channels,
                            kernel_size=2,
                            stride=2,
                            padding=0,
                            bias=False,
                            mode="saab",
                            name="hop4_hop1_ch"
                            + str(hop1_index)
                            + "_hop2_ch"
                            + str(hop2_index)
                            + "_hop3_ch"
                            + str(hop3_index),
                        )
                    )
                self.cw_conv4[hop1_index][hop2_index] = nn.ModuleList(
                    self.cw_conv4[hop1_index][hop2_index]
                )
            self.cw_conv4[hop1_index] = nn.ModuleList(self.cw_conv4[hop1_index])
        self.cw_conv4 = nn.ModuleList(self.cw_conv4)

    def forward(self, input):
        with torch.no_grad():
            output1 = self.conv1(input)
            if self.training:
                self.eng1 = self.conv1.explained_variance_ratio_

            output2 = []
            if self.training:
                self.eng2 = []
            for hop1_index, conv2 in enumerate(self.cw_conv2):
                output2.append(conv2(output1[:, hop1_index].unsqueeze(1)))
                if self.training:
                    self.eng2.append(
                        conv2.explained_variance_ratio_ * self.eng1[hop1_index]
                    )

            output3 = []
            if self.training:
                self.eng3 = []
            for hop1_index, hop2_conv in enumerate(self.cw_conv3):
                output3.append([])
                if self.training:
                    self.eng3.append([])
                for hop2_index, conv3 in enumerate(hop2_conv):  # type: ignore
                    output3[hop1_index].append(
                        conv3(output2[hop1_index][:, hop2_index].unsqueeze(1))
                    )
                    if self.training:
                        self.eng3[hop1_index].append(
                            conv3.explained_variance_ratio_
                            * self.eng2[hop1_index][hop2_index]
                        )

            output4 = []
            if self.training:
                self.eng4 = []
            for hop1_index, hop2_conv in enumerate(self.cw_conv4):
                output4.append([])
                if self.training:
                    self.eng4.append([])
                for hop2_index, hop3_conv in enumerate(hop2_conv):  # type: ignore
                    output4[hop1_index].append([])
                    if self.training:
                        self.eng4[hop1_index].append([])
                    for hop3_index, conv4 in enumerate(hop3_conv):
                        output4[hop1_index][hop2_index].append(
                            conv4(
                                output3[hop1_index][hop2_index][
                                    :, hop3_index
                                ].unsqueeze(1)
                            )
                        )
                        if self.training:
                            self.eng4[hop1_index][hop2_index].append(
                                conv4.explained_variance_ratio_
                                * self.eng3[hop1_index][hop2_index][hop3_index]
                            )
        return output1, output2, output3, output4

    def inverse4(self, output4):
        with torch.no_grad():
            inv4 = []
            for hop1_index, hop2_conv in enumerate(self.cw_conv4):
                inv4.append([])
                for hop2_index, hop3_conv in enumerate(hop2_conv):  # type: ignore
                    inv4[hop1_index].append([])
                    for hop3_index, conv4 in enumerate(hop3_conv):
                        if (
                            hop1_index < len(output4)
                            and hop2_index < len(output4[hop1_index])
                            and hop3_index < len(output4[hop1_index][hop2_index])
                        ):
                            inv4[hop1_index][hop2_index].append(
                                conv4.inverse(
                                    output4[hop1_index][hop2_index][hop3_index]
                                )
                            )
                    inv4[hop1_index][hop2_index] = torch.cat(
                        inv4[hop1_index][hop2_index], dim=1
                    )
            inv1 = self.inverse3(inv4)
        return inv1

    def inverse3(self, output3):
        with torch.no_grad():
            inv3 = []
            for hop1_index, hop2_conv in enumerate(self.cw_conv3):
                inv3.append([])
                for hop2_index, conv3 in enumerate(hop2_conv):  # type: ignore
                    if hop1_index < len(output3) and hop2_index < len(
                        output3[hop1_index]
                    ):
                        inv3[hop1_index].append(
                            conv3.inverse(output3[hop1_index][hop2_index])
                        )
                inv3[hop1_index] = torch.cat(inv3[hop1_index], dim=1)
            inv1 = self.inverse2(inv3)
        return inv1

    def inverse2(self, output2):
        with torch.no_grad():
            inv2 = []
            for hop1_index, conv2 in enumerate(self.cw_conv2):
                if hop1_index < len(output2):
                    inv2.append(conv2.inverse(output2[hop1_index]))  # type: ignore
            inv2 = torch.cat(inv2, dim=1)
            inv1 = self.inverse1(inv2)
        return inv1

    def inverse1(self, output1):
        with torch.no_grad():
            inv1 = self.conv1.inverse(output1)
        inv1 = torch.clamp(inv1, min=0, max=1)
        return inv1

    def concat(self, output, energy):
        return self._concat(output, energy, "")

    def _concat(self, output, energy, name):
        if isinstance(output, torch.Tensor):
            names_list = []
            for idx in range(output.shape[1]):
                if idx == 0:
                    names_list.append(name + "DC")
                    offset = 0
                else:
                    if idx == 0:
                        offset = 1
                    names_list.append(name + f"AC{idx+offset}")
            return output, energy, names_list
        if isinstance(output, list):
            output_list = []
            names_list = []
            energy_list = []
            for idx, o in enumerate(output):
                if idx == 0:
                    out, eng, names = self._concat(o, energy[idx], name + "DC ")
                    offset = 0
                else:
                    if idx == 0:
                        offset = 1
                    out, eng, names = self._concat(
                        o, energy[idx], name + f"AC{idx+offset} "
                    )
                output_list.append(out)
                energy_list.append(eng)
                names_list += names
            return (
                torch.cat(output_list, dim=1),
                torch.cat(energy_list, dim=0),
                names_list,
            )
