import os
from xml.dom import minidom
import glob

xmls = glob.glob("themes/*.xml")

file = open("themes.txt", "w")

for xml in xmls:
    mydoc = minidom.parse(xml)
    items = mydoc.getElementsByTagName('color')
    color = items[0].firstChild.data
    h = color.lstrip('#')
    color_rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    file.write(xml.split("\\")[1] + ": "+str(color_rgb)+"\n")

file.close()
