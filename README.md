# ✨ Air Drawing Application Using Hand Gestures

A real-time virtual drawing tool that uses hand gestures to draw, erase, change brush color, and clear the screen — all without touching any device!

---

## 📌 Features

- 🖌️ **Brush & Eraser** tool controlled by hand gestures  
- 🎨 **Color selection** using on-screen color swatches  
- ✋ **Palm gesture** to clear the canvas  
- 👉 **Index finger** tracking to draw on screen  
- 👆 **Thumb gestures** to switch tools (Brush / Eraser)

---

## 🧠 Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**

---

## 🔧 How It Works

- The app uses your webcam to detect your hand using MediaPipe.
- Recognizes gestures (e.g., thumbs up, palm open, one finger up) to trigger tools.
- Tracks the index fingertip to draw lines on a transparent canvas.
- Combines canvas and webcam view for a seamless experience.


---

## 📂 Folder Structure

project/
│
├── hand.py # Main application file
└── README.md # Project documentation
