import os
import torch
import torch.nn as nn
import numpy as np
from models.transformer import TransformerBlock
from utils import LearnableSigmoid2d
from pesq import pesq
from joblib import Parallel, delayed

import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out
    

class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DenseBlock(h, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super(TSTransformerBlock, self).__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class GatedFusionLayer(nn.Module):
    def __init__(self, condition_dim, feature_channel):
        super(GatedFusionLayer, self).__init__()
        self.feature_channel = feature_channel

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_channel, feature_channel),
            nn.ReLU()
        )

        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, feature_channel),
            nn.ReLU()
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(feature_channel * 2, feature_channel),
            nn.Sigmoid()
        )

    def forward(self, x, E_cond):
        B, C, T, F = x.shape

        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, T*F, C)
        x_proj = self.feature_proj(x_flat)

        e_proj = self.condition_proj(E_cond)
        e_proj = e_proj.unsqueeze(1).expand(-1, T*F, -1)

        gate_input = torch.cat([x_proj, e_proj], dim=-1)
        gate = self.gate_proj(gate_input)

        fused = x_proj * (1 - gate) + e_proj * gate
        fused = fused.view(B, T, F, C).permute(0, 3, 1, 2)
        
        return fused

class ECAPAExtractorModule(nn.Module):
    def __init__(self, model_path, device_id=0):
        super().__init__()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.sample_rate = 16000

    def forward(self, wav):
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        
        device = wav.device
        batch_size = wav.shape[0]

        all_feats = []
        wav_scaled = wav * (1 << 15) 
        for i in range(batch_size):
            feat = kaldi.fbank(
                wav_scaled[i:i+1],
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                sample_frequency=self.sample_rate,
                window_type='hamming',
                use_energy=False
            )
            feat = feat - torch.mean(feat, dim=0)
            all_feats.append(feat.cpu())
        feats = torch.stack(all_feats).cpu().numpy()  # [B, T, F]

        embeddings = self.session.run(
            output_names=[self.output_name],
            input_feed={self.input_name: feats}
        )[0]

        return torch.from_numpy(embeddings).to(device)

class InstantiationModule(nn.Module):
    def __init__(self, h, embed_dir, K_components, D_features):
        super(InstantiationModule, self).__init__()
        self.K = K_components
        self.D = D_features

        mu_clean = torch.load(os.path.join(embed_dir, 'clean_gmm_mu_k.pt')).float()
        self.register_buffer('mu_clean', mu_clean) 
        self.temperature = h.get('temperature', 0.05)

    def forward(self, spk_emb):
        Q_norm = torch.nn.functional.normalize(spk_emb, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(self.mu_clean, p=2, dim=-1)

        scores = torch.matmul(Q_norm, mu_norm.t()) / self.temperature

        gamma = torch.nn.functional.softmax(scores, dim=-1)

        # if torch.rand(1).item() < 0.01:
        #     max_val, max_idx = torch.max(gamma, dim=1)
        #     print(f"| GMM Geometry Match: Idx={max_idx[0].item()}, Prob={max_val[0].item():.4f}")

        E_instantiated = torch.matmul(gamma, self.mu_clean)
        
        return E_instantiated

class MPNet(nn.Module):
    def __init__(self, h, num_tsblocks=4):
        super(MPNet, self).__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.embed_dim = self.h.embed_dim 
        self.C = self.h.dense_channel 
        self.K = self.h.K_components 

        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.ecapa_extractor = ECAPAExtractorModule(
            model_path=h.ecapa_model_path, 
            device_id=0
        )

        self.instantiation_module = InstantiationModule(
            h, h.gmm_embed_dir, h.K_components, self.embed_dim
        )

        self.gated_fusion = GatedFusionLayer(condition_dim=192, feature_channel=self.C)

        self.TSTransformer = nn.ModuleList([TSTransformerBlock(h) for _ in range(num_tsblocks)])
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_wav, noisy_amp, noisy_pha):
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)
        x = self.dense_encoder(x) 

        with torch.no_grad():
            spk_emb = self.ecapa_extractor(noisy_wav)
            spk_emb = spk_emb.detach()

        E_inst = self.instantiation_module(spk_emb) 

        x = self.gated_fusion(x, E_inst) 

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)
        
        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack((denoised_amp*torch.cos(denoised_pha),
                                    denoised_amp*torch.sin(denoised_pha)), dim=-1)

        return denoised_amp, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1

    return pesq_score