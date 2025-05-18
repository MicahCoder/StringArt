from PIL import Image, ImageDraw
import numpy as np
import csv
from math import pi, sin, cos, sqrt,dist
from rich.progress import track

image = Image.open("inputs/kahlo.png")
# weight = Image.open("inputs/einstienWeights.png").convert("L")

bwImage = image.convert("L")
image = image.convert("RGBA")
overlay = Image.new('RGBA', image.size, (0,0,0,0))
#Define Image Array
arr = np.array(bwImage).astype(np.float32)
# arrweight = np.array(weight).astype(np.float32)
width,height = arr.shape[0], arr.shape[1]

draw = ImageDraw.Draw(overlay, "RGBA")
start_point = (0, 0)
end_point = (width, height)
alpha = 0.125
removeScore = 1.25
line_color = (0, 0, 0,int(255*alpha))
line_width = 1

def drawPoint(x, y):
    # Draw a point on the image
    draw.ellipse(xy=(x-2, y-2, x + 2, y + 2), fill=(0, 0, 0))
center = (width // 2, height // 2)

numPoints = 180
circleScale = 0.99
points = [(center[0] + width/2*circleScale* cos(2*pi/numPoints*i), center[1]+ height/2*circleScale* sin(2*pi/numPoints*i)) for i in range(numPoints)]


for point in points:
    drawPoint(point[0], point[1])

# draw.line([start_point, end_point], fill=line_color, width=line_width)
def drawLine(start, end,overlay):
    # Draw a line on the image
    overlay1 = Image.new('RGBA', image.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay1, "RGBA")
    draw.line([start, end], fill=line_color, width=line_width)
    return Image.alpha_composite(overlay, overlay1)

def loss(start, end):
    if start == end:
        return float("inf")
    end,start = max(start, end), min(start, end)
    steps = int(dist(start, end))
    total = 0.0
    x0, y0 = start
    dx = (end[0] - x0) / steps
    dy = (end[1] - y0) / steps
    for i in range(steps):
        x0 += dx
        y0 += dy
        # drawPoint(x, y)
        total += arr[int(y0)][int(x0)]#*arrweight[int(y0)][int(x0)]/255
    return total/steps
def minLoss(startIndex, points):
    minLoss = float("inf")
    minPoint = None
    start = points[startIndex]
    for i in range(len(points)):
        point = points[i]
        l = loss(start, point)
        if l < minLoss:
            minLoss = l
            minPoint = i
    return minPoint
def lowerArray(start, end):
    end,start = max(start, end), min(start, end)
    steps = int(sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2))
    total = 0.0
    x0, y0 = start
    x1, y1 = end
    for i in range(steps):
        x = x0 + (x1 - x0) * i / steps
        y = y0 + (y1 - y0) * i / steps
        arr[int(y)][int(x)] *= removeScore
        
steps = 4000
point = 0

instructions = []

for i in track(range(steps)):
    nextPoint = minLoss(point, points)
    instructions.append((point, nextPoint))
    pointVal = points[point]
    nextPointVal = points[nextPoint]
    overlay = drawLine(pointVal, nextPointVal, overlay= overlay)
    lowerArray(pointVal, nextPointVal)
    point = nextPoint
image = Image.alpha_composite(image, overlay)
# image = overlay
# Convert to RGB before saving
foldername = "kahlo"
image.save("savedOutputs/"+foldername+"/render.png")
overlay.save("savedOutputs/"+foldername+"output/overlay.png")
overlay.show()
image.show()
with open("savedOutputs/"+foldername+"output/instructions.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow("Point #")
    writer.writerows(instructions)

