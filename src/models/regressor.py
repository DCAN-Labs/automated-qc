from monai.networks.nets import Regressor

from models.temporal import FrameWiseRegressor

DEFAULT_CHANNELS = (16, 32, 64, 128, 256, 512, 1024)
DEFAULT_STRIDES = (2, 2, 2, 2, 2, 2)


def get_regressor_model(
    spatial_shape=(260, 320, 320),
    in_channels=1,
    channels=DEFAULT_CHANNELS,
    strides=DEFAULT_STRIDES,
    frame_mode=None,
    temporal_pool="mean",
):
    """Build the MONAI Regressor, optionally wrapped for 4D input.

    Args:
        spatial_shape: the 3 spatial dims the network is built for.
        in_channels: conv input channels. For frame_mode="channels" this is the
            frame count; otherwise 1.
        channels / strides: MONAI Regressor layer configuration.
        frame_mode: None or "3d" for the original behaviour; "channels" to feed
            frames as input channels; "pool" to encode each frame with a shared
            3D backbone and pool the per-frame predictions.
        temporal_pool: reduction used when frame_mode="pool".

    Historical configurations, for reference:
        models 00-05: in_shape=[1, 260, 320, 320], strides=(2,2,2,2,2,2)
        model 06+:    in_shape=[1, 194, 232, 158], strides=(2,2,2,2,1,1,1)
    If you change the spatial shape here, change --target-shape to match.
    """
    in_shape = [in_channels, *spatial_shape]

    print("=" * 50)
    print("CREATING NEW REGRESSOR MODEL")
    print(f"in_shape: {in_shape}")
    print(f"channels: {channels}")
    print(f"strides: {strides}")
    print(f"frame_mode: {frame_mode or '3d'}")
    if frame_mode == "pool":
        print(f"temporal_pool: {temporal_pool}")
    print("=" * 50)

    model = Regressor(
        in_shape=in_shape,
        out_shape=1,
        channels=tuple(channels),
        strides=tuple(strides),
    )

    if frame_mode == "pool":
        model = FrameWiseRegressor(model, pool=temporal_pool)

    print(f"Model created: {model}")

    return model
