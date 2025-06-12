#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 12:01:21 2025

@author: georgieauvray
"""
import matplotlib.pyplot as plt

## FUNCTIONS TO CREATE THE PITCH
def create_pitch(length=100, width=68, end=10, figsize=(7,5), lwd=1):
    """Create a rugby pitch using matplotlib"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Pitch outline
    plt.plot([0, 0, length, length, 0], [0, width, width, 0, 0], color="black", linewidth=1.5*lwd)
    ax.plot([0, -end, -end, 0], [0, 0, width, width], color="black", linewidth=lwd)
    ax.plot([length, length+end, length+end, length], [0, 0, width, width], color="black", linewidth=1)

    # other lines
    x = [length/2, 22, length-22, # half and 22m
         5, 5, 5, 5, 5, 5, # 5m lines
         length-5, length-5, length-5, length-5, length-5, length-5, 
         length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10,# 10m lines
         length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10,
         0, 0, 0, 0, length, length, length, length, # vertical 5m
         19.5, 19.5, 19.5, 19.5, length-19.5, length-19.5, length-19.5, length-19.5, # vertical 22m
         length/2-12.5, length/2-12.5, length/2-12.5, length/2-12.5, # vertical 10m
         length/2+12.5, length/2+12.5, length/2+12.5, length/2+12.5,
         length/2-2.5, length/2-2.5, length/2-2.5, length/2-2.5, # vertical centre
         length/2-0.5, -0.5, -0.5, length-0.5, length-0.5] # goal and centre dash
    y = [0, 0, 0, # half and 22m
         2.5, 12.5, width-2.5, width-12.5, width/2-2.8, width/2+2.8,  # 5m lines
         2.5, 12.5, width-2.5, width-12.5, width/2-2.8, width/2+2.8,
         2.5, 12.5, 22.5, width-2.5, width-12.5, width-22.5, width/2-2.5, # 10m lines
         2.5, 12.5, 22.5, width-2.5, width-12.5, width-22.5, width/2-2.5,
         5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 5m
         5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 22m
         5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 10m
         5, 15, width-5, width-15, # centre
         width/2, width/2-2.8, width/2+2.8, width/2-2.8, width/2+2.8] # goal and centre dash
    endx = [length/2, 22, length-22, # half and 22m
         5, 5, 5, 5, 5, 5, # 5m lines
         length-5, length-5, length-5, length-5, length-5, length-5, 
         length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, # 10m lines
         length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10,
         5, 5, 5, 5, length-5, length-5, length-5, length-5, # vertical 5m
         24.5, 24.5, 24.5, 24.5, length-24.5, length-24.5, length-24.5, length-24.5, # vertical 22m
         length/2-7.5, length/2-7.5, length/2-7.5, length/2-7.5, # vertical 10m
         length/2+7.5, length/2+7.5, length/2+7.5, length/2+7.5,
         length/2+2.5, length/2+2.5, length/2+2.5, length/2+2.5, # vertical centre
         length/2+0.5, 0.5, 0.5, length+0.5, length+0.5] # goal and centre dash
    endy = [width, width, width, # half and 22m
            7.5, 17.5, width-7.5, width-17.5, width/2-7.8, width/2+7.8, # 5m lines
            7.5, 17.5, width-7.5, width-17.5, width/2-7.8, width/2+7.8,
            7.5, 17.5, 27.5, width-7.5, width-17.5, width-27.5, width/2+2.5, # 10m lines
            7.5, 17.5, 27.5, width-7.5, width-17.5, width-27.5, width/2+2.5,
            5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 5m
            5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 22m
            5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 10m
            5, 15, width-5, width-15, # centre
            width/2, width/2-2.8, width/2+2.8, width/2-2.8, width/2+2.8] # goal and centre dash
    
    for x, y, endx, endy in zip(x,y,endx,endy):
        ax.plot([x, endx], [y, endy], color="black", linewidth=lwd)
    
    # Tidy axes
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax

def create_rotated_pitch(length=100, width=68, end=10, figsize=(5,7), lwd=1):
    """Create a rugby pitch using matplotlib"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Pitch outline
    plt.plot([0, width, width, 0, 0], [0, 0, length, length, 0], color="black", linewidth=1.5*lwd)

    # other lines
    x = [length/2, 22, length-22, # half and 22m
         5, 5, 5, 5, 5, 5, # 5m lines
         length-5, length-5, length-5, length-5, length-5, length-5, 
         length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10,# 10m lines
         length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10,
         0, 0, 0, 0, length, length, length, length, # vertical 5m
         19.5, 19.5, 19.5, 19.5, length-19.5, length-19.5, length-19.5, length-19.5, # vertical 22m
         length/2-12.5, length/2-12.5, length/2-12.5, length/2-12.5, # vertical 10m
         length/2+12.5, length/2+12.5, length/2+12.5, length/2+12.5,
         length/2-2.5, length/2-2.5, length/2-2.5, length/2-2.5, # vertical centre
         length/2-0.5, -0.5, -0.5, length-0.5, length-0.5] # goal and centre dash
    y = [0, 0, 0, # half and 22m
         2.5, 12.5, width-2.5, width-12.5, width/2-2.8, width/2+2.8,  # 5m lines
         2.5, 12.5, width-2.5, width-12.5, width/2-2.8, width/2+2.8,
         2.5, 12.5, 22.5, width-2.5, width-12.5, width-22.5, width/2-2.5, # 10m lines
         2.5, 12.5, 22.5, width-2.5, width-12.5, width-22.5, width/2-2.5,
         5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 5m
         5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 22m
         5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 10m
         5, 15, width-5, width-15, # centre
         width/2, width/2-2.8, width/2+2.8, width/2-2.8, width/2+2.8] # goal and centre dash
    endx = [length/2, 22, length-22, # half and 22m
         5, 5, 5, 5, 5, 5, # 5m lines
         length-5, length-5, length-5, length-5, length-5, length-5, 
         length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, length/2-10, # 10m lines
         length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10, length/2+10,
         5, 5, 5, 5, length-5, length-5, length-5, length-5, # vertical 5m
         24.5, 24.5, 24.5, 24.5, length-24.5, length-24.5, length-24.5, length-24.5, # vertical 22m
         length/2-7.5, length/2-7.5, length/2-7.5, length/2-7.5, # vertical 10m
         length/2+7.5, length/2+7.5, length/2+7.5, length/2+7.5,
         length/2+2.5, length/2+2.5, length/2+2.5, length/2+2.5, # vertical centre
         length/2+0.5, 0.5, 0.5, length+0.5, length+0.5] # goal and centre dash
    endy = [width, width, width, # half and 22m
            7.5, 17.5, width-7.5, width-17.5, width/2-7.8, width/2+7.8, # 5m lines
            7.5, 17.5, width-7.5, width-17.5, width/2-7.8, width/2+7.8,
            7.5, 17.5, 27.5, width-7.5, width-17.5, width-27.5, width/2+2.5, # 10m lines
            7.5, 17.5, 27.5, width-7.5, width-17.5, width-27.5, width/2+2.5,
            5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 5m
            5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 22m
            5, 15, width-5, width-15, 5, 15, width-5, width-15, # vertical 10m
            5, 15, width-5, width-15, # centre
            width/2, width/2-2.8, width/2+2.8, width/2-2.8, width/2+2.8] # goal and centre dash
    
    for x, y, endx, endy in zip(x,y,endx,endy):
        ax.plot([y, endy], [x, endx], color="black", linewidth=lwd)
    
    # Tidy axes
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax
