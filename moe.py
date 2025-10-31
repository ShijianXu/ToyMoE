import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts with sparse routing."""

    def __init__(self, num_experts, gates: torch.Tensor):
        """
        Args:
            num_experts: int
            gates: [B, E] tensor, with at most k nonzeros per row
        """
        self._gates = gates
        self._num_experts = num_experts

        # 取 gate>0 的 (batch_idx, expert_idx) 索引，确保 as_tuple=False 以得到 [N,2]
        indices = torch.nonzero(gates > 0, as_tuple=False)  # [N, 2]
        if indices.numel() == 0:
            # 极端情况：所有 gates 都是 0（理论上 top-k 不会出现），兜底处理
            device = gates.device
            self._expert_index = torch.empty(0, dtype=torch.long, device=device)
            self._batch_index = torch.empty(0, dtype=torch.long, device=device)
            self._part_sizes = [0] * num_experts
            self._nonzero_gates = torch.empty(0, 1, dtype=gates.dtype, device=device)
            return

        # indices 的第 0 列是 batch_idx，第 1 列是 expert_idx
        expert_idx_col = indices[:, 1]
        # 按专家索引排序，这样相同专家的样本连续分组
        sorted_experts, sort_idx = expert_idx_col.sort()  # both [N]
        self._expert_index = sorted_experts  # 每个非零 gate 对应的专家 id（按专家分组后的顺序）
        self._batch_index = indices[sort_idx, 0]  # 对应的 batch 索引

        # 每个专家接收的样本数（part sizes）
        self._part_sizes = (gates > 0).sum(0).tolist()  # len = num_experts

        # 为了后面结合输出时加权，需要取出每个非零 gate 的值（与 _batch_index/_expert_index 对齐）
        gates_expanded = gates[self._batch_index]  # [N, E]
        # gather 需要 index 的 shape 与输出一致，这里取每行对应专家的列
        self._nonzero_gates = gates_expanded.gather(1, self._expert_index.unsqueeze(1))  # [N,1]

    def dispatch(self, inp: torch.Tensor):
        """
        为每个专家创建一个输入张量（将属于该专家的样本挑出，按专家分组）
        Args:
            inp: [B, D]
        Returns:
            list of length num_experts; 第 i 个元素形状 [expert_batch_size_i, D]
        """
        # 根据 _batch_index 重排 inp，然后按 _part_sizes 切分
        # 注意：不要 squeeze(1)，inp[self._batch_index] 已是 [N, D]
        inp_expanded = inp[self._batch_index]
        return torch.split(inp_expanded, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates: bool = True):
        """
        将专家输出按照 gates 加权、按 batch 位置聚合回去
        Args:
            expert_out: list，len=num_experts，第 i 个是 [expert_batch_size_i, O]
            multiply_by_gates: 是否按 gate 权重加权
        Returns:
            combined: [B, O]
        """
        if len(expert_out) == 0:
            # 兜底：无样本
            B = self._gates.size(0)
            return torch.zeros(B, 0, device=self._gates.device)

        stitched = torch.cat(expert_out, dim=0)  # [N, O]
        if multiply_by_gates:
            # _nonzero_gates 是 [N,1]，可广播到输出维
            stitched = stitched * self._nonzero_gates

        B = self._gates.size(0)
        O = stitched.size(1)
        # 不需要 requires_grad=True，这只是汇聚容器
        combined = torch.zeros(B, O, device=stitched.device, dtype=stitched.dtype)
        # 把各片段按 _batch_index 累加回原位置
        combined.index_add_(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        """返回与每个专家输入一一对应的 gate 权重（按专家分组后的顺序）。"""
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class ToyMoE(nn.Module):
    def __init__(self, in_channels, input_size, output_size, hidden_size, num_experts, noisy_gating=True, k=2):
        super().__init__()
        assert k <= num_experts, "k must be <= num_experts"

        self.noisy_gating = noisy_gating
        self.in_channels = in_channels
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k  # top-k experts

        # 1) 特征提取
        self.extractor = ConvNet(in_channels)

        # 2) 专家网络
        self.experts = nn.ModuleList(
            [MLP(self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)]
        )

        # 3) gating 参数（改为 Xavier 初始化，避免全 0 打不破对称）
        self.w_gate = nn.Parameter(torch.empty(input_size, num_experts))
        self.w_noise = nn.Parameter(torch.empty(input_size, num_experts))
        nn.init.xavier_uniform_(self.w_gate)
        nn.init.xavier_uniform_(self.w_noise)

        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    @staticmethod
    def cv_squared(x: torch.Tensor):
        """CV^2 = Var/Mean^2，鼓励均匀分布。空张量或单元素返回 0."""
        if x.numel() <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        eps = 1e-10
        return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)

    @staticmethod
    def _gates_to_load(gates: torch.Tensor):
        """每个专家被分到的样本数量（>0 的个数）"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        计算每个值在加入噪声后落入 top-k 的概率（用于可导的负载估计）
        Shapes:
          clean_values: [B, E]
          noisy_values: [B, E]
          noise_stddev: [B, E]
          noisy_top_values: [B, m], m >= k+1
        """
        B = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.reshape(-1)

        # top-k 阈值（in 与 out 两种情况下的相邻阈值）
        base = torch.arange(B, device=clean_values.device) * m
        threshold_positions_if_in = base + self.k
        threshold_if_in = top_values_flat.index_select(0, threshold_positions_if_in).unsqueeze(1)

        is_in = noisy_values > threshold_if_in

        threshold_positions_if_out = base + self.k - 1
        threshold_if_out = top_values_flat.index_select(0, threshold_positions_if_out).unsqueeze(1)

        normal = Normal(self.mean, self.std)
        # 避免除 0
        noise_stddev = torch.clamp(noise_stddev, min=1e-6)

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2):
        """
        Noisy Top-K Gating（Shazeer et al., 2017）
        返回：
          gates: [B, E] 稀疏矩阵（每行仅 top-k 非零，且行内归一化和为 1）
          load:  [E] 每个专家的“负载”（可微估计或计数）
        """
        # logits: [B, E]
        clean_logits = x @ self.w_gate  # 先不 softmax

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            used_for_topk = noisy_logits
        else:
            noise_stddev = None
            used_for_topk = clean_logits

        # 先 top-k（以及多取一个以便概率估计）
        topk = min(self.k + 1, self.num_experts)
        top_logits, top_indices = used_for_topk.topk(topk, dim=1)
        top_k_logits = top_logits[:, :self.k]                # [B, k]
        top_k_indices = top_indices[:, :self.k]              # [B, k]

        # 对 top-k 内做 softmax，得到行内归一化的 gate 值
        top_k_gates = F.softmax(top_k_logits, dim=1)         # [B, k]

        # 将稀疏 gate 写回到 [B, E]
        gates = torch.zeros_like(clean_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)

        # 负载估计：训练时用概率估计，可微；否则用计数
        if self.noisy_gating and train and (self.k < self.num_experts):
            load = self._prob_in_top_k(clean_logits, used_for_topk, noise_stddev, top_logits).sum(0)
            load = load.float()
        else:
            load = self._gates_to_load(gates).float()

        return gates, load

    def forward(self, x, loss_coef: float = 1e-2):
        """
        返回:
          y:   [B, output_size]
          aux: 标量辅助损失（负载均衡）
        """
        # 1) 特征提取
        x = self.extractor(x)     # [B, input_size]

        # 2) 计算 gates 和 load
        gates, load = self.noisy_top_k_gating(x, self.training)

        # 3) 负载均衡损失（importance + load 的 CV^2）
        importance = gates.sum(0)                      # [E]
        aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        aux_loss = aux_loss * loss_coef

        # 4) 分派样本给专家
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)         # list of [n_i, D]

        # 5) 专家计算
        expert_outputs = []
        for i in range(self.num_experts):
            if expert_inputs[i].numel() == 0:
                # 零样本专家，构造空输出（保持 shape 一致）
                expert_outputs.append(
                    torch.zeros(0, self.output_size, device=x.device, dtype=x.dtype)
                )
            else:
                expert_outputs.append(self.experts[i](expert_inputs[i]))  # [n_i, O]

        # 6) 合并
        y = dispatcher.combine(expert_outputs, multiply_by_gates=True)  # [B, O]
        return y, aux_loss



class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, output_dim):
        super(SimpleConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)