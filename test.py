
colors = {
    "30": [0,0,0],
    "31": [205,0,0],
    "32": [0,205,0],
    "33": [205,205,0],
    "34": [0,0,238],
    "35": [205,0,205],
    "36": [0,205,205],
    "37": [229,229,229],
    "90": [127,127,127],
    "91": [255,0,0],
    "92": [0,255,0],
    "93": [255,255,0],
    "94": [92,92,255],
    "95": [255,0,255],
    "96": [0,255,255],
    "97": [255,255,255]
    }


pixel = [214, 90, 5]

def find_color(pixel):
    closest = [0,0,0]
    closest_dist = 255*3
    for color in colors:
        dist = 0
        for i in range(3):
            dist += abs(pixel[i] - colors[color][i])
        if dist < closest_dist:
            closest_dist = dist
            closest = color
    return closest

