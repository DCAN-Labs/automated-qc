from monai.networks.nets import Regressor


def get_regressor_model():
    
    # models 00-05 use default parameters of
    # in_shape=[1, 260, 320, 320]
    # channels=(16, 32, 64, 128, 256, 512, 1024)
    # strides=(2, 2, 2, 2, 2, 2)
    
    # model 06+ has been using custom parameters
    # in_shape=[1, 194, 232, 158]
    # channels=(16, 32, 64, 128, 256, 512, 1024)
    # strides=(2, 2, 2, 2, 1, 1, 1)
     
    in_shape=[1, 260, 320, 320]
    channels=(16, 32, 64, 128, 256, 512, 1024)
    strides=(2, 2, 2, 2, 2, 2)
    
     # Print to verify what's being used
    print("=" * 50)
    print("CREATING NEW REGRESSOR MODEL")
    print(f"in_shape: {in_shape}")
    print(f"channels: {channels}")
    print(f"strides: {strides}")
    print("=" * 50)
    
    model = Regressor(
        in_shape=in_shape,
        out_shape=1,
        channels=channels,
        strides=strides
    )

    print(f"Model created: {model}")
    
    return model
