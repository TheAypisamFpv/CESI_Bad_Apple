import time


class Style:
    BLOCK = 0
    DOT_GRID = 1

def getProgressBar(completion: float, wheelIndex=None, style=None, maxbarLength=75):
    """
    Generate a progress bar with optional style selection.
    
    Args:
        completion (float): Progress completion between 0.0 and 1.0
        wheelIndex (int): Optional index for progress wheel animation
        style (int): Optional style parameter, defaults to Style.BLOCK
        maxbarLength (int): Maximum length of the progress bar
    
    Returns:
        str: Formatted progress bar string
    """

    completion = min(max(completion, 0.0), 1.0)
    # Default style to BLOCK if None
    if style is None:
        style = Style.BLOCK
    
    # ANSI color codes
    ORANGE = "\033[93m"  # Orange color
    GRAY = "\033[90m"    # Dark gray color
    RESET = "\033[0m"    # Reset to default color

    completionPercent = f"{completion * 100:.2f}"

    # Define progress wheel animation characters (shared between styles)
    progressWheel = ["◜", "◝", "◞", "◟"]
    if wheelIndex is None:
        # Default to a wheel index based on completion
        wheelIndex = int(completion * 4 * maxbarLength) % len(progressWheel)
    
    wheelChar = progressWheel[wheelIndex % len(progressWheel)]

    if completion == 1.0:
        wheelChar = "-"
    
    if style == Style.BLOCK:  # BLOCK style
        fullBlocks = int(completion * maxbarLength)
        partialBlock = (completion * maxbarLength - fullBlocks)
        bar = "█" * fullBlocks
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
        bar = bar.ljust(maxbarLength)
        return f"Progress: [{bar}] {completionPercent:>6}%  {wheelChar}  "
    
    elif style == Style.DOT_GRID:  # DOT_GRID style
        maxbarLength = maxbarLength + 2
        filledDots = round(completion * maxbarLength)
        # Construct bar with colored colons
        progressBarPart = ORANGE + ":" * filledDots
        backgroundPart = GRAY + ":" * (maxbarLength - filledDots)
        bar = progressBarPart + backgroundPart + RESET

        return f"Progress: {bar} {completionPercent:>6}%  {wheelChar}  "

# Example usage
# if __name__ == "__main__":
#     # Default block style
#     # for i in range(501):
#     #     progress = i / 500
#     #     print(getProgressBar(progress, wheelIndex=i), end='\r')
#     #     time.sleep(0.03)

#     # print()

#     # Dot grid style
#     for i in range(501):
#         progress = i / 500
#         print(getProgressBar(progress, wheelIndex=i, style=Style.BLOCK), end='\r')
#         time.sleep(0.03)

#     print()