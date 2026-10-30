"""Reject unsupported normalization configuration without performing it."""


def validate_normalization_config(data):
    if type(data.use_quantile_norm) is not bool or getattr(data, "standard_norm_keys", ()):
        raise ValueError("unsupported normalization: require Standard or Quantile; mixed normalization unsupported")
