def getProgressBar(completion:float, wheelIndex=None, maxbarLength=20):
    """Generate a progress bar and progress wheel for the given completion percentage.

    This function creates a progress bar with a specified number of blocks and partial blocks
    based on the completion percentage. It uses Unicode block characters to represent the progress.

    Args:
        completion (float): The percentage of completion (0 to 1).
        maxbarLength (int, optional): The total number of blocks in the progress bar. Defaults to 20.

    Returns:
        str: A string representing the progress bar with Unicode block characters.
    """
    completionPercent = f"{completion * 100:.2f}"
    fullBlocks = int(completion * maxbarLength)
    partialBlock = (completion * maxbarLength - fullBlocks)
    bar = "█" * fullBlocks

    # Use appropriate Unicode characters for partial blocks
    if partialBlock > 0:
        if partialBlock < 0.125:
            bar += "▏"
        elif partialBlock < 0.25:
            bar += "▎"
        elif partialBlock < 0.375:
            bar += "▍"
        elif partialBlock < 0.5:
            bar += "▌"
        elif partialBlock < 0.625:
            bar += "▋"
        elif partialBlock < 0.75:
            bar += "▊"
        elif partialBlock < 0.875:
            bar += "▉"
        else:
            bar += "█"

    progressWheel = ["|", "/", "-", "\\"]
    # use partial block to determine which character to display
    if not wheelIndex:
        wheelIndex = int(partialBlock * 4)
    
    wheelChar = progressWheel[wheelIndex % len(progressWheel)]

    bar = bar.ljust(maxbarLength)
    return f"Progress: [{bar}] {completionPercent:>6}%  {wheelChar}  "