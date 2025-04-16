#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import shutil

def change_file_extension_with_shutil(src_filename, dst_filename):
    shutil.move(src_filename, dst_filename)

def random_pacman_starting_position(layout_dir_name, string_suffix):
    layout_names = os.listdir(layout_dir_name)

    for file_name in layout_names:
        # has to end in .lay but shouldn't be one of the new maps we're making
        if file_name[-6:] == "_1.lay":
            with open(f"{layout_dir_name}{file_name}") as f:
                with open(f"{layout_dir_name}{file_name[:-4]}{string_suffix}.txt", 'w') as output_file:
                    lines = f.readlines()
                    row_max = len(lines)
                    col_max = len(lines[0])
            
                    # get rid of pacman
                    lines = [line.replace("P", " ") for line in lines]
            
                    # loop until we find a valid new location for pacman
                    while True:
                        x_coord = np.random.randint(1, col_max - 1)
                        y_coord = np.random.randint(1, row_max - 1)
                        if lines[y_coord][x_coord] == " ":
                            lines[y_coord] = lines[y_coord][:x_coord] + "P" + lines[y_coord][x_coord+1:]
                            break
            
                    for line in lines:
                        output_file.write(line)

                change_file_extension_with_shutil(f"{layout_dir_name}{file_name[:-4]}{string_suffix}.txt",
                                                  f"{layout_dir_name}{file_name[:-4]}{string_suffix}.lay")


if __name__ == "__main__":

    dir_name = "./layouts_copy/"

    random_pacman_starting_position(dir_name, "_2")
    random_pacman_starting_position(dir_name, "_3")
    random_pacman_starting_position(dir_name, "_4")
    random_pacman_starting_position(dir_name, "_5")

    layout_names = os.listdir(dir_name)

    for file_name in layout_names:
        # has to end in .lay but shouldn't be one of the new maps we're making
        if file_name[-4:] == ".lay" and file_name[-6] != "_":
            shutil.move(f"{dir_name}{file_name}",
                        f"{dir_name}{file_name[:-4]}_1.lay")




