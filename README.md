# 🚗 Car Parking Space Detection System

This is an *automated parking space detection system* built using **OpenCV, **cvzone, **NumPy, and **Python*. It identifies free and occupied parking spaces in a video feed and provides real-time updates on parking availability. The system is interactive and allows users to define parking spaces on an image of a parking lot, which are then tracked to detect vehicle occupancy.

## 🛠 Tech Stack

### 🖼 Computer Vision

- *OpenCV*: The primary library for image processing, used to detect free and occupied parking spots by analyzing video frames.
- *cvzone*: An easy-to-use wrapper for OpenCV that helps with adding visual elements, like text overlays and rectangles, on the frames.
- *NumPy*: Used for efficient handling of pixel data and manipulating the frames.

### 📊 Detection Logic

- *Parking Space Picker*: An interactive tool that allows users to select parking spaces manually by clicking on an image of the parking lot. Parking spaces are stored in a file for future use.
- *Real-Time Analysis*: The system continuously analyzes a video feed to determine which parking spaces are occupied and which are available, updating the count in real-time.

### ⚙ DevOps

- *Pickle Files*: The coordinates of the parking spaces are saved as serialized pickle files, which are loaded into memory during the detection phase for quick processing.

## 🚀 Features

- *🖱 Interactive Parking Space Selection*: Users can manually select parking spaces by clicking on an image of the parking lot. The selected positions are stored in a file for future detection.
- *📹 Real-Time Detection*: The system processes a video feed of the parking lot and detects which parking spaces are free or occupied.
- *🔧 Adjustable Thresholds*: Users can dynamically adjust the detection sensitivity using trackbars, allowing for better accuracy under different lighting conditions or video qualities.

## 🛠 How It Works

### Parking Space Picker

The *ParkingSpacePicker.py* script allows you to manually select parking spaces on an image of the parking lot by clicking on it. The coordinates of the selected spaces are saved for use in the real-time detection.

- *Left Click*: Add a parking space by clicking on the image.
- *Right Click*: Remove a parking space by right-clicking within its boundaries.

### Real-Time Detection

The *main.py* script analyzes a video feed to detect whether each parking space is free or occupied. It processes each frame of the video, counts the number of non-zero pixels within the parking space boundaries, and classifies the spot as free or occupied based on a predefined threshold.

## 📦 Running the Application Locally

### Prerequisites

- *Python 3.x*: Make sure Python is installed on your machine.
- *OpenCV and NumPy*: Install the required libraries via pip:


### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ErsatzHitman/car-parking-detection-system.git
   cd car-parking-detection-system
2. **Run the Parking Space Picker file**:

    ```bash
    python ParkingSpacePicker.py
    ```
3. **Run Real-Time Detection**:
   ```bash
    python main.py
    ```
Now watch the real-time detection of parking spaces in the video feed.

## 🎥 Demo

Check out the video demo [here](https://drive.google.com/file/d/1ORBdzU42iiSwNFZTy0C6A8OnaHUJIrSx/view?usp=sharing).

## 📸 Screenshots

Here are a few screenshots of the application:

![Parking Space Selection](https://github.com/user-attachments/assets/24bc8701-a5ac-43d2-8d6b-1f9ba3f15591)
![Real-Time Detection](https://github.com/user-attachments/assets/0111d751-323a-4e5d-bcf1-3a007e2c7867)


