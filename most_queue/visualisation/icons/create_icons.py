from PIL import Image
import os


def get_colors():
    colors = {}
    with open("themes.txt", 'r', encoding='utf-8') as file:
        for line in file:
            lines = line.split(":")
            colors[lines[0].strip()] = lines[1].strip()

    return colors


def create_icons_from_folder(folder_path, folder_color):
    colors = get_colors()

    tek_dir = os.getcwd()

    out_path = os.path.join(tek_dir, "icons")

    print(out_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for theme in colors:

        print("Theme {0:s}".format(theme))

        theme_folder = os.path.join(out_path, theme.split(".")[0])

        if not os.path.exists(theme_folder):
            os.makedirs(theme_folder)

        icons = os.listdir(folder_path)

        for icon in icons:

            print("---- icon: {0:s}".format(icon))
            icon_path = os.path.join(folder_path, icon)
            im = Image.open(icon_path)
            pixelMap = im.load()

            img = Image.new(im.mode, im.size)
            pixelsNew = img.load()
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    if pixelMap[i, j] == folder_color:
                        color = colors[theme].replace("(", "")
                        color = color.replace(")", "")
                        color_mass = color.split(',')
                        new_color = (int(color_mass[0]), int(color_mass[1]), int(color_mass[2]), 255)
                        pixelMap[i, j] = new_color
                    else:
                        pixelMap[i, j] = (0, 0, 0, 0)
                    pixelsNew[i, j] = pixelMap[i, j]

            img_name = icon
            img_name = os.path.join(theme_folder, img_name)
            img.save(img_name)
            img.close()

            im.close()


def create_icons(path_to_source_icon, source_color_rgba):
    colors = get_colors()

    tek_dir = os.getcwd()

    out_path = os.path.join(tek_dir, "icons")

    print(out_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for theme in colors:

        im = Image.open(path_to_source_icon)
        pixelMap = im.load()

        img = Image.new(im.mode, im.size)
        pixelsNew = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixelMap[i, j] == source_color_rgba:
                    color = colors[theme].replace("(", "")
                    color = color.replace(")", "")
                    color_mass = color.split(',')
                    new_color = (int(color_mass[0]), int(color_mass[1]), int(color_mass[2]), 255)
                    pixelMap[i, j] = new_color
                else:
                    pixelMap[i, j] = (0, 0, 0, 0)
                pixelsNew[i, j] = pixelMap[i, j]

        img_name = theme.split(".")[0] + ".png"
        img_name = os.path.join(out_path, img_name)
        img.save(img_name)
        img.close()

        im.close()


if __name__ == "__main__":
    # create_icons('play-button.png', )
    # create_icons('maps.png', (0,0,0, 255))
    create_icons_from_folder("dark_amber", (255, 215, 64, 255))
