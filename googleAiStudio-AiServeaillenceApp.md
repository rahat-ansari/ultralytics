# VisionGuard AI - Intelligent Surveillance System

**VisionGuard AI** is a state-of-the-art security dashboard that transforms standard media feeds into actionable intelligence. By leveraging the **Google Gemini Multimodal API**, it simulates a professional security suite—integrating Object Detection (YOLO), Identity Tracking (DeepSORT), and Face Verification—without requiring heavy local machine learning models.

This application is designed for **Home Security** and **Livestock Monitoring**, providing real-time threat assessment, intruder detection, and automated alerts.

## 🚀 Key Features

### 🧠 Next-Gen AI Analysis

- **Smart Face Cropping**: Automatically detects faces or estimates head positions (even when faces are obscured) to generate centered, high-visibility avatars for the UI.
- **DeepSORT Simulation**: Maintains consistent "Tracking IDs" for subjects across video frames, understanding movement context and occlusion.
- **Face Verification**: Compares detected individuals against a user-managed "Known Persons" database.

### 🛡️ Operational Intelligence

- **Adaptive Video Pacing**: The video analysis engine automatically throttles requests based on API latency and rate limits to ensure stability without crashing.
- **Granular Object Classification**: Goes beyond generic labels—identifying specific car models, animal breeds, and object colors (e.g., "Red Tactical Backpack", "2018 Toyota Camry").
- **Asset Protection**: Monitors proximity to user-defined critical assets (Safes, Entrances, Livestock Pens).

### ⚡ Interactive & Responsive

- **Night Mode (IR Enhance)**: Applies client-side Histogram Equalization to simulate Infrared Night Vision, improving detection in low-light images.
- **Multi-Sensory Alerts**:
  - **Visual**: Color-coded threat banners and bounding box overlays.
  - **Audio**: Synthesized siren utilizing the Web Audio API.
  - **Voice**: Text-to-Speech announcements for critical High-Threat events.

---

## 📥 Download & Installation

You can get the source code up and running on your local machine in minutes.

### Prerequisites

- **Node.js** (v18.0.0 or higher)
- **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/))

### Option 1: Clone via Git (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/visionguard-ai.git

# 2. Navigate to the project directory
cd visionguard-ai

# 3. Install dependencies
npm install
```

### Option 2: Download ZIP

1.  Click the **Code** button at the top of the repository.
2.  Select **Download ZIP**.
3.  Extract the ZIP file.
4.  Open the folder in your terminal and run `npm install`.

---

## ⚙️ Configuration

1.  **Set up Environment Variables**:
    Create a `.env` file in the root directory:

    ```env
    # Linux/Mac
    touch .env

    # Windows
    type nul > .env
    ```

2.  **Add your API Key**:
    Open `.env` and paste your key:

    ```env
    API_KEY=AIzaSy...YourActualKeyHere
    ```

    _(Note: Depending on your build tool, you may need to use `VITE_API_KEY` and update the service file accordingly)._

3.  **Run the App**:
    ```bash
    npm run dev
    ```
    Access the app at `http://localhost:5173`.

---

## 📖 User Guide

### 1. Mode Selection

- **Home Security**: Focuses on "Intruders" vs "Family". High sensitivity to unknown persons near entrances.
- **Cattle Shelter**: Focuses on "Predators" (Wolves, stray dogs) vs "Livestock" (Cows, Sheep).

### 2. Managing Known Personnel

- Use the **Control Panel** to upload photos of authorized individuals.
- **Tip**: Use clear, well-lit headshots for the best accuracy.
- You can upload individual files or import an entire folder.

### 3. Media Analysis

- **Upload**: Drag & drop images or video files (`.mp4`, `.webm`).
- **Webcam**: Switch to "Live Camera" mode to snap real-time photos for analysis.
- **Settings**: Adjust the _Confidence Threshold_ slider. Higher values (e.g., 0.85) require a stricter face match; lower values (e.g., 0.60) are more lenient.

---

## 🛠️ Technology Stack

- **Frontend**: React 19, TypeScript, Vite
- **UI/UX**: Tailwind CSS, Lucide React Icons
- **AI Engine**: Google GenAI SDK (`@google/genai`)
- **Data Persistence**: LocalStorage (for history and settings)

## ⚠️ Limitations & Privacy

- **Rate Limits**: This app relies on the Google Gemini API. Video processing is intensive; free-tier keys may hit rate limits. The app includes auto-retry logic to mitigate this.
- **Data Privacy**: Media data is sent to Google's servers for analysis. Ensure you comply with local surveillance laws.
