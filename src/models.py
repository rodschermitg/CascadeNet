import torch
import torch.nn as nn


def match_to(x, ref, keep_axes=(1,)):
    target_shape = list(ref.shape)
    for i in keep_axes:
        target_shape[i] = x.shape[i]
    target_shape = tuple(target_shape)

    if x.shape == target_shape:
        pass
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        while x.dim() < len(target_shape):
            x = x.unsqueeze(-1)

    x = x.expand(*target_shape)
    x = x.to(device=ref.device, dtype=ref.dtype)

    return x


def is_conv(op):
    conv_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d
    )

    if type(op) is type and issubclass(op, conv_types):
        return True
    elif type(op) in conv_types:
        return True
    else:
        return False


class MixtureOfGaussians(torch.distributions.MixtureSameFamily, torch.distributions.Distribution):
    has_rsample = True

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):

        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        assert self._component_distribution.has_rsample is True

        if not isinstance(self._mixture_distribution, torch.distributions.RelaxedOneHotCategorical):
            raise ValueError("The Mixture distribution needs to be an instance of torch.distributions.Distribution.RelaxedOneHotCategorical")

        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]

        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError(
                    "`mixture_distribution.batch_shape` ({0}) is not "
                    "compatible with `component_distribution'."
                    "batch_shape`({1})".format(mdbs, cdbs)
                )

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km
        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)

        torch.distributions.Distribution.__init__(
            self,
            batch_shape=cdbs,
            event_shape=event_shape,
            validate_args=validate_args
        )

    def get_mixture_distribution(self):
        return self.mixture_distribution

    def get_component_distribution(self):
        return self.component_distribution

    def rsample(self, sample_shape=torch.Size(), show_indices=False):

        # Shape : [B, n_components]
        mix_sample = self.mixture_distribution.rsample(sample_shape)

        # Straight-Through Gumble-Softmax: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
        # See also: https://wiki.lzhbrian.me/notes/differientiable-sampling-and-argmax
        # See: E. Jang, S. Gu, and B. Poole. Categorical Reparameterization with Gumbel-Softmax (2017), ICLR 2017
        # In the forward-pass, we get the one-hot vector (the actual sample cancels out) and in the backward pass
        # argmax does not have a contribute to the gradient but mix_sample does!
        index = mix_sample.max(dim=-1, keepdim=True)[1]
        sample_mask = torch.zeros_like(mix_sample).scatter_(-1, index, 1.0)

        # Shape: [B, n_components]
        sample_mask = sample_mask - mix_sample.detach() + mix_sample

        # Shape: [B, n_components, z_dim]
        comp_samples = self.component_distribution.rsample(sample_shape)

        # Add a "fake" axis to the sample mask to broadcast the multiplication
        samples = torch.mul(comp_samples, sample_mask.unsqueeze(dim=-1))

        # The one-hot sample mask will zero-out all the rows
        # apart from the "selected" mixture component
        # So by summing along the columns, we recover the
        # sample from the "winning" Gaussian
        samples = torch.sum(samples, dim=-2)

        if show_indices is False:
            return samples
        else:
            return torch.cat([samples, index], dim=-1)


class FullCovConvGaussianEncoder(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with a full covariance matrix.
    """
    def __init__(
        self,
        in_channels,
        latent_size,
        encoder_op,
        encoder_kwargs=None,
        depth=5,
        block_depth=2,
        num_feature_maps=24,
        feature_map_multiplier=2,
        n_components=1
    ):
        super(FullCovConvGaussianEncoder, self).__init__()

        self.default_encoder_kwargs = {
            "in_channels": in_channels,
            "depth": depth,
            "block_depth": block_depth,
            "num_feature_maps": num_feature_maps,
            "feature_map_multiplier": feature_map_multiplier
        }

        self.in_channels = in_channels
        self.num_filters = num_feature_maps * (feature_map_multiplier ** (depth-1))
        self.latent_size = latent_size
        self.n_components = n_components

        self.encoder_op = encoder_op
        self.encoder_kwargs = self.default_encoder_kwargs.copy()
        if encoder_kwargs is not None:
            self.encoder_kwargs.update(encoder_kwargs)

        self.encoder = self.encoder_op(**self.encoder_kwargs)

        # Obtain the following from encoder output: mu, log(sigma), L'
        self.mean_log_sigma_op = nn.Conv3d(self.num_filters, self.n_components*2*self.latent_size, (1, 1, 1), stride=1)

        # Lower triangular part of the covariance matrix
        self.cov_tril_op = nn.Conv3d(self.num_filters, self.n_components*self.latent_size*self.latent_size, (1, 1, 1), stride=1)

        if self.n_components > 1:
            self.mixture_weights_conv = nn.Conv3d(self.num_filters, self.n_components, (1, 1, 1), stride=1)

    def forward(self, x):
        encoding = self.encoder(x)

        mu_log_sigma = self.mean_log_sigma_op(encoding)
        mu_log_sigma = mu_log_sigma.squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)
        mu_log_sigma = mu_log_sigma.view(mu_log_sigma.shape[0], self.n_components, 2*self.latent_size)

        cov_tril = self.cov_tril_op(encoding)
        cov_tril = cov_tril.squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)

        # Shape: [B*n_components x latent_size x latent_size]
        cov_tril = cov_tril.view(cov_tril.shape[0]*self.n_components, self.latent_size, self.latent_size)

        # Get lower triangular part (without the diagonal)
        L_hat = torch.tril(cov_tril, diagonal=-1)

        mu = mu_log_sigma[:, :, :self.latent_size]
        log_sigma = mu_log_sigma[:, :, self.latent_size:]

        # Shape: [B*n_components x latent_size]
        log_sigma = log_sigma.view(mu_log_sigma.shape[0]*self.n_components, self.latent_size)

        # See Pg 29 of D. P. Kingma and M. Welling, An Introduction to Variational Autoencoders, FNT in Machine Learning, vol. 12, no. 4, pp. 307–392, 2019, doi: 10.1561/2200000056.
        # See also: https://discuss.pytorch.org/t/operation-on-diagonals-of-matrix-batch/50779
        L = L_hat.clone()
        # Add the diagonal elements (sigma) to L with noise for numerical stability
        L.diagonal(dim1=-2, dim2=-1)[:] += (torch.exp(log_sigma) + 0.0001)

        # Reshape L
        L = L.view(mu_log_sigma.shape[0], self.n_components, self.latent_size, self.latent_size)

        if self.n_components == 1:
            mixture_weights = None
        else:
            mixture_weights = self.mixture_weights_conv(encoding)  # Shape : [batch_size, n_components, 1, 1, 1]
            mixture_weights = mixture_weights.squeeze((-3, -2, -1))

        return (L, mu, mixture_weights)


class ConvModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvModule, self).__init__()

    def init_weights(self, init_fn, *args, **kwargs):
        class init_(object):
            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if is_conv(type(module)):
                    module.weight = self.fn(module.weight, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)

    def init_bias(self, init_fn, *args, **kwargs):
        class init_(object):
            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if is_conv(type(module)) and module.bias is not None:
                    module.bias = self.fn(module.bias, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)


class InjectionConvEncoder(ConvModule):
    def __init__(
        self,
        in_channels,
        depth,
        block_depth,
        num_feature_maps,
        feature_map_multiplier,
        activation_op=nn.LeakyReLU,
        activation_kwargs=None,
        norm_op=nn.InstanceNorm3d,
        norm_kwargs=None,
        norm_depth=0,
        conv_op=nn.Conv3d,
        conv_kwargs=None,
        pool_op=nn.AvgPool3d,
        pool_kwargs=None,
        dropout_op=None,
        dropout_kwargs=None,
        global_pool_op=nn.AdaptiveAvgPool3d,
        global_pool_kwargs=None,
        **kwargs
    ):
        super(InjectionConvEncoder, self).__init__(**kwargs)

        self.default_activation_kwargs = {"inplace": True}
        self.default_norm_kwargs = {}
        self.default_conv_kwargs = {"kernel_size": 3, "padding": 1}
        self.default_pool_kwargs = {"kernel_size": 2}
        self.default_dropout_kwargs = {}
        self.default_global_pool_kwargs = {}

        self.in_channels = in_channels
        self.depth = depth
        self.block_depth = block_depth
        self.num_feature_maps = num_feature_maps
        self.feature_map_multiplier = feature_map_multiplier

        self.activation_op = activation_op
        self.activation_kwargs = self.default_activation_kwargs.copy()
        if activation_kwargs is not None:
            self.activation_kwargs.update(activation_kwargs)

        self.norm_op = norm_op
        self.norm_kwargs = self.default_norm_kwargs.copy()
        if norm_kwargs is not None:
            self.norm_kwargs.update(norm_kwargs)
        self.norm_depth = depth if norm_depth == "full" else norm_depth

        self.conv_op = conv_op
        self.conv_kwargs = self.default_conv_kwargs.copy()
        if conv_kwargs is not None:
            self.conv_kwargs.update(conv_kwargs)

        self.pool_op = pool_op
        self.pool_kwargs = self.default_pool_kwargs.copy()
        if pool_kwargs is not None:
            self.pool_kwargs.update(pool_kwargs)

        self.dropout_op = dropout_op
        self.dropout_kwargs = self.default_dropout_kwargs.copy()
        if dropout_kwargs is not None:
            self.dropout_kwargs.update(dropout_kwargs)

        self.global_pool_op = global_pool_op
        self.global_pool_kwargs = self.default_global_pool_kwargs.copy()
        if global_pool_kwargs is not None:
            self.global_pool_kwargs.update(global_pool_kwargs)

        for d in range(self.depth):
            in_ = self.in_channels if d == 0 else self.num_feature_maps * (self.feature_map_multiplier**(d-1))
            out_ = self.num_feature_maps * (self.feature_map_multiplier**d)

            layers = []
            if d > 0:
                layers.append(self.pool_op(**self.pool_kwargs))
            for b in range(self.block_depth):
                current_in = in_ if b == 0 else out_
                layers.append(self.conv_op(current_in, out_, **self.conv_kwargs))
                if self.norm_op is not None and d < self.norm_depth:
                    layers.append(self.norm_op(out_, **self.norm_kwargs))
                if self.activation_op is not None:
                    layers.append(self.activation_op(**self.activation_kwargs))
                if self.dropout_op is not None:
                    layers.append(self.dropout_op(**self.dropout_kwargs))

            self.add_module("encode_{}".format(d), nn.Sequential(*layers))

        if self.global_pool_op is not None:
            self.add_module("global_pool", self.global_pool_op(1, **self.global_pool_kwargs))

    def forward(self, x):
        for d in range(self.depth):
            x = self._modules["encode_{}".format(d)](x)

        if hasattr(self, "global_pool"):
            x = self.global_pool(x)

        return x


class InjectionUNet(ConvModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        injection_channels,
        depth=5,
        block_depth=2,
        num_feature_maps=24,
        feature_map_multiplier=2,
        kernel_size=3,
        dilation=1,
        num_1x1_at_end=3,
        activation_op=nn.LeakyReLU,
        activation_kwargs=None,
        pool_op=nn.AvgPool3d,
        pool_kwargs=None,
        dropout_op=None,
        dropout_kwargs=None,
        norm_op=nn.InstanceNorm3d,
        norm_kwargs=None,
        conv_op=nn.Conv3d,
        conv_kwargs=None,
        upconv_op=nn.ConvTranspose3d,
        upconv_kwargs=None,
        output_activation_op=None,
        output_activation_kwargs=None,
        **kwargs
    ):
        super(InjectionUNet, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.injection_channels = injection_channels
        self.depth = depth
        self.block_depth = block_depth
        self.num_feature_maps = num_feature_maps
        self.feature_map_multiplier = feature_map_multiplier
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (self.kernel_size + (self.kernel_size-1) * (self.dilation-1)) // 2
        self.num_1x1_at_end = num_1x1_at_end
        self.activation_op = activation_op
        self.activation_kwargs = {"inplace": True} if activation_kwargs is None else activation_kwargs
        self.pool_op = pool_op
        self.pool_kwargs = {"kernel_size": 2} if pool_kwargs is None else pool_kwargs
        self.dropout_op = dropout_op
        self.dropout_kwargs = {} if dropout_kwargs is None else dropout_kwargs
        self.norm_op = norm_op
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_op = conv_op
        self.conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        self.upconv_op = upconv_op
        self.upconv_kwargs = {} if upconv_kwargs is None else upconv_kwargs
        self.output_activation_op = output_activation_op
        self.output_activation_kwargs = {} if output_activation_kwargs is None else output_activation_kwargs

        self.last_activations = None

        # BUILD ENCODER
        for d in range(self.depth):
            block = []
            if d > 0:
                block.append(self.pool_op(**self.pool_kwargs))

            for i in range(self.block_depth):
                # bottom block fixed to have depth 1
                if d == self.depth - 1 and i > 0:
                    continue

                out_size = self.num_feature_maps * feature_map_multiplier**d
                if d == 0 and i == 0:
                    in_size = self.in_channels
                elif i == 0:
                    in_size = self.num_feature_maps * feature_map_multiplier**(d - 1)
                else:
                    in_size = out_size

                block.append(self.conv_op(
                    in_size,
                    out_size,
                    self.kernel_size,
                    padding=self.padding,
                    dilation=self.dilation,
                    **self.conv_kwargs
                ))
                if self.dropout_op is not None:
                    block.append(self.dropout_op(**self.dropout_kwargs))
                if self.norm_op is not None:
                    block.append(self.norm_op(out_size, **self.norm_kwargs))
                block.append(self.activation_op(**self.activation_kwargs))

            self.add_module("encode-{}".format(d), nn.Sequential(*block))

        # BUILD DECODER
        for d in reversed(range(self.depth)):
            block = []

            for i in range(self.block_depth):
                # bottom block fixed to have depth 1
                if d == self.depth - 1 and i > 0:
                    continue

                out_size = self.num_feature_maps * feature_map_multiplier**(d)
                if i == 0 and d < self.depth - 1:
                    in_size = self.num_feature_maps * feature_map_multiplier**(d+1)
                else:
                    in_size = out_size

                block.append(self.conv_op(
                    in_size,
                    out_size,
                    self.kernel_size,
                    padding=self.padding,
                    dilation=self.dilation,
                    **self.conv_kwargs
                ))
                if self.dropout_op is not None:
                    block.append(self.dropout_op(**self.dropout_kwargs))
                if self.norm_op is not None:
                    block.append(self.norm_op(out_size, **self.norm_kwargs))
                block.append(self.activation_op(**self.activation_kwargs))

            if d > 0:
                block.append(self.upconv_op(
                    out_size,
                    out_size // 2,
                    self.kernel_size,
                    2,
                    padding=self.padding,
                    dilation=self.dilation,
                    output_padding=1,
                    **self.upconv_kwargs
                ))

            self.add_module("decode-{}".format(d), nn.Sequential(*block))

        out_size += self.injection_channels
        in_size = out_size
        for i in range(self.num_1x1_at_end):
            if i == self.num_1x1_at_end - 1:
                out_size = self.out_channels
            current_conv_kwargs = self.conv_kwargs.copy()
            current_conv_kwargs["bias"] = True
            self.add_module("reduce-{}".format(i), self.conv_op(in_size, out_size, 1, **current_conv_kwargs))
            if i != self.num_1x1_at_end - 1:
                self.add_module("reduce-{}-nonlin".format(i), self.activation_op(**self.activation_kwargs))
        if self.output_activation_op is not None:
            self.add_module("output-activation", self.output_activation_op(**self.output_activation_kwargs))

    def reset(self):
        self.last_activations = None

    def forward(
        self,
        x,
        injection=None,
        reuse_last_activations=False,
        store_activations=False
    ):
        if self.last_activations is None or reuse_last_activations is False:
            enc = [x]

            for i in range(self.depth - 1):
                enc.append(self._modules["encode-{}".format(i)](enc[-1]))

            bottom_rep = self._modules["encode-{}".format(self.depth - 1)](enc[-1])
            x = self._modules["decode-{}".format(self.depth - 1)](bottom_rep)

            for i in reversed(range(self.depth - 1)):
                x = self._modules["decode-{}".format(i)](torch.cat((enc[-(self.depth - 1 - i)], x), 1))

            if store_activations:
                self.last_activations = x.detach()

        else:
            x = self.last_activations

        if self.injection_channels > 0:
            injection = match_to(injection, x, (0, 1))
            x = torch.cat((x, injection), 1)

        for i in range(self.num_1x1_at_end):
            x = self._modules["reduce-{}".format(i)](x)
        if self.output_activation_op is not None:
            x = self._modules["output-activation"](x)

        return x


class ProbabilisticSegmentationNet(ConvModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_size,
        temperature=0.1,
        task_op=InjectionUNet,
        task_kwargs=None,
        prior_op=FullCovConvGaussianEncoder,
        prior_kwargs=None,
        posterior_op=FullCovConvGaussianEncoder,
        posterior_kwargs=None,
        **kwargs
    ):
        super(ProbabilisticSegmentationNet, self).__init__(**kwargs)

        self.default_task_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "injection_channels": latent_size
        }
        self.default_prior_kwargs = {
            "in_channels": in_channels,
            "latent_size": latent_size,
            "encoder_op": InjectionConvEncoder,
            "encoder_kwargs": {"norm_depth": "full"}
        }
        self.default_posterior_kwargs = {
            "in_channels": in_channels+out_channels,
            "latent_size": latent_size,
            "encoder_op": InjectionConvEncoder,
            "encoder_kwargs": {"norm_depth": "full"}
        }

        self.task_op = task_op
        self.task_kwargs = self.default_task_kwargs.copy()
        if task_kwargs is not None:
            self.task_kwargs.update(task_kwargs)

        self.prior_op = prior_op
        self.prior_kwargs = self.default_prior_kwargs.copy()
        if prior_kwargs is not None:
            self.prior_kwargs.update(prior_kwargs)

        self.posterior_op = posterior_op
        self.posterior_kwargs = self.default_posterior_kwargs.copy()
        if posterior_kwargs is not None:
            self.posterior_kwargs.update(posterior_kwargs)

        self.temperature = temperature
        self._prior = None
        self._posterior = None

        self.make_modules()

    def make_modules(self):
        self.add_module("task_net", self.task_op(**self.task_kwargs))
        self.add_module("prior_net", self.prior_op(**self.prior_kwargs))
        self.add_module("posterior_net", self.posterior_op(**self.posterior_kwargs))

    @property
    def prior(self):
        return self._prior

    @property
    def posterior(self):
        return self._posterior

    @property
    def last_activations(self):
        return self.task_net.last_activations

    def train(self, mode=True):
        super(ProbabilisticSegmentationNet, self).train(mode)
        self.reset()

    def reset(self):
        self.task_net.reset()
        self._prior = None
        self._posterior = None

    def forward(self, input_, seg=None):
        """Forward pass includes reparametrization sampling during training, otherwise it'll just take the prior mean."""
        self.encode_prior(input_)
        if self.training:
            self.encode_posterior(input_, seg)
            sample = self.posterior.rsample()
        else:
            sample = self.prior.mean
        reconstruction = self.task_net(input_, sample, store_activations=not self.training)

        if self.training:
            kld = self.compute_kl_divergence(posterior_dist=self.posterior,
                                             prior_dist=self.prior)
            kld = torch.mean(kld)

            return (reconstruction, kld)
        else:
            return reconstruction

    def encode_latent_dist(self, scale_tril, loc, logits):
        if logits is None:
            scale_tril = scale_tril.squeeze(dim=1)
            loc = loc.squeeze(dim=1)
            dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        else:
            cat_distribution = torch.distributions.RelaxedOneHotCategorical(
                logits=logits,
                temperature=torch.Tensor([self.temperature]).to(logits.device)
            )
            comp_distribution = torch.distributions.MultivariateNormal(
                loc=loc,
                scale_tril=scale_tril
            )
            dist = MixtureOfGaussians(
                mixture_distribution=cat_distribution,
                component_distribution=comp_distribution
            )
        return dist

    def encode_prior(self, input_):
        (scale_tril, loc, logits) = self.prior_net(input_)
        latent_distribution = self.encode_latent_dist(scale_tril, loc, logits)
        self._prior = latent_distribution
        return self._prior

    def encode_posterior(self, input_, seg):
        (scale_tril, loc, logits) = self.posterior_net(torch.cat((input_, seg.float()), 1))
        latent_distribution = self.encode_latent_dist(scale_tril, loc, logits)
        self._posterior = latent_distribution
        return self._posterior

    @staticmethod
    def compute_kl_divergence(posterior_dist, prior_dist, mc_samples=1000):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """

        try:
            # Need to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = torch.distributions.kl_divergence(posterior_dist, prior_dist)

        except NotImplementedError:
            # If the analytic KL divergence does not exists, use MC-approximation
            # See: 'APPROXIMATING THE KULLBACK LEIBLER DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS' by Hershey and Olsen (2007)

            # MC-approximation
            posterior_samples = posterior_dist.rsample(sample_shape=torch.Size([mc_samples]))
            log_posterior_prob = posterior_dist.log_prob(posterior_samples)
            log_prior_prob = prior_dist.log_prob(posterior_samples)
            monte_carlo_terms = log_posterior_prob - log_prior_prob
            kl_div = torch.mean(monte_carlo_terms, dim=0)

        return kl_div
