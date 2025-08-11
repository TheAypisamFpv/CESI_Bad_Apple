class Style:
    BLOCK = 0
    DOT_GRID = 1

def getProgressBar(completion: float, wheelIndex: int=None, style: int=Style.BLOCK, maxbarLength: int=50) -> str:
    """
    Generate a progress bar with optional style selection.
    
    Args:
        completion (float): Progress completion between 0.0 and 1.0
        wheelIndex (int): Optional index for progress wheel animation
        style (int): Optional style parameter, defaults to Style.BLOCK
        maxbarLength (int): Optional maximum length of the progress bar
    
    Returns:
        str: Formatted progress bar string
    
    ## Usage:
        Always use with carriage return when printing to create an updating effect:
        print(getProgressBar(0.5), end='\\r')
        
        Example in a loop:
    ```
    for i, item in enumerate(items):
        completion = i / len(items)
        print(getProgressBar(completion, wheelIndex=i), end='\\r')
    ```
            
    ## Note:
        Includes a space character at the end of the progress bar.
    """

    completion = min(max(completion, 0.0), 1.0)
    
    # Default style to BLOCK if None
    if style not in Style.__dict__.values():
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
        progressBar = "█" * fullBlocks
        if partialBlock > 0:
            if partialBlock < 0.125:
                progressBar += "▏"
            elif partialBlock < 0.25:
                progressBar += "▎"
            elif partialBlock < 0.375:
                progressBar += "▍"
            elif partialBlock < 0.5:
                progressBar += "▌"
            elif partialBlock < 0.625:
                progressBar += "▋"
            elif partialBlock < 0.75:
                progressBar += "▊"
            elif partialBlock < 0.875:
                progressBar += "▉"
            else:
                progressBar += "█"
                
        progressBar = f"[{progressBar.ljust(maxbarLength)}]"
        
    elif style == Style.DOT_GRID:  # DOT_GRID style
        maxbarLength = maxbarLength + 2
        filledDots = round(completion * maxbarLength)
        # Construct bar with colored colons
        progressBarPart = ORANGE + ":" * filledDots
        backgroundPart = GRAY + ":" * (maxbarLength - filledDots)
        
        progressBar = progressBarPart + backgroundPart + RESET

    return f"Progress: {progressBar} {completionPercent:>6}%  {wheelChar}  "


# Example usage

# Uncomment to see examples in action
# if __name__ == "__main__":
#     import time
#   
#     # Default block style example
#     print("Example with default block style:")
#     for i in range(1001):
#         progress = i / 1000
#         print(getProgressBar(progress, wheelIndex=i), end='\r')
#         time.sleep(0.01)
#     print("\n")
#   
#     # Dot grid style example
#     print("Example with dot grid style:")
#     for i in range(1001):
#         progress = i / 1000
#         print(getProgressBar(progress, style=Style.DOT_GRID), end='\r')
#         time.sleep(0.01)
#     print("\n")