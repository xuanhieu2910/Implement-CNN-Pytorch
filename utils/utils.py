
def print_train_time(start_time: float, end_time: float, device = None):
    """
    Print training time prints time between start and end time training
    :arg
        start_time [float]: start time
        end_time [float]: end time
        device [Optional: None]: device
    :return:
        Total training time
    """
    total_time = end_time - start_time
    print(f"Training time: {total_time: .3f} seconds")
    return total_time


def load_config():
    return f"Load config!"