# **Wav2Lip Project**


# Video Resume Generator 


**Goal:** Generate a fully automated, high-quality, and personalized **Visual Resume Video** of a candidate using inputs from a mock interview (video, audio, transcript). The final video should simulate the candidate speaking confidently using their own voice and a rewritten script.

---

## Overview

This project uses a combination of **GenAI**, **voice cloning**, **deepfake video generation**, and **automated video processing** tools to create a professional-looking **video resume** from an interview recording — all done through code. No manual editing is involved.

---

##  Input

-  Interview Recording (Video + Audio)
-  Transcript of Interview
-  Still image or frame of candidate (optional for SadTalker/Wav2Lip)

---

##  Pipeline

```mermaid
graph TD;
    A[Interview: Video + Audio + Transcript] --> B[Transcript → GPT-3.5 → Self-intro Script];
    A --> C[Extract Voice → Clone Voice from Interview Audio];
    B --> C2[Script + Cloned Voice → AI-generated Audio];
    C2 --> D[Generate Lip-synced Deepfake Video];
    D --> E[Enhance with Background, Clothes, Duration (MoviePy/FFmpeg/DeepMotion)];
    E --> F[Final Visual Resume Output];
````

---

##  Detailed Workflow

### 1.  Transcript Rewriting

* Use a **language model** like `ChatGPT` (Free / GPT-3.5) to rewrite the original transcript into a **clean, structured self-introduction**.
* The script highlights:

  * Candidate's background
  * Skills and experience
  * Career goals

> *Example prompt*: “Rewrite this interview transcript as a 60-second self-introduction for a video resume.”

---

### 2.  Voice Cloning

* Extract the candidate’s voice characteristics using the original interview audio.
* Clone voice using tools like:

  * [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) (offline, open-source)
  * [Resemble.ai](https://www.resemble.ai/) (API, paid)
  * [ElevenLabs](https://www.elevenlabs.io/) (API, freemium)
* Input: self-introduction script
* Output: cloned audio (.wav/.mp3)

---

### 3.  Deepfake Video Generation

* Choose from:

  * **SadTalker** – Full-face animation
  * **Wav2Lip** – Lip-sync only using a static image or frame
  * **DeepMotion** – Optional full-body motion animation based on audio and transcript
* Sync cloned audio with the image or video to generate a natural-looking video.

---

### 4.  Post-processing (Optional Enhancements)

* Use tools like:

  * `FFmpeg` or `MoviePy` to merge video, audio, and background
  * `RemBG` to remove and replace backgrounds
  * `Stable Diffusion` or `AI Clothes Swap` to digitally change attire
  * **DeepMotion** to animate a full-body 3D avatar (if required)

---

### 5.  Output

* A final, clean **Visual Resume Video** (.mp4) that:

  * Sounds like the candidate
  * Reflects a confident self-introduction
  * Includes a professional background and custom visuals
  * Is generated entirely through code

---

##  Tools & Libraries Used

| Purpose             | Tool / Library                          |
| ------------------- | --------------------------------------- |
| Script Rewriting    | OpenAI ChatGPT (Free/GPT-3.5)           |
| Voice Cloning       | Tortoise TTS / ElevenLabs               |
| Lip Sync Video      | Wav2Lip                                 |
| Full Body Animation | DeepMotion (optional)                   |
| Background/Attire   | RemBG, Stable Diffusion                 |
| Stitching & Editing | FFmpeg                                  |

---

##  Folder Structure

```bash
.
├── input/
│   ├── interview_video.mp4
│   ├── transcript.txt
│   └── candidate_photo.jpg
├── scripts/
│   ├── 1_transcript_to_script.py
│   ├── 2_voice_cloning.py
│   ├── 3_generate_video.py
│   ├── 4_postprocess.py
├── output/
│   └── final_visual_resume.mp4
├── checkpoints/
│   └── wav2lip_gan.pth
├── assets/
│   └── architecture_diagram.png
├── README.md
└── requirements.txt
```

---

##  Sample Output

> A sample generated resume video will appear after processing is complete.

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone http://github.com/irapadole15/visual-resume-generator.git
cd visual-resume-generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download pretrained models

* Download `s3fd.pth` (face detector) → `face_detection/sfd/`
* Download `wav2lip_gan.pth` → `checkpoints/`

### 4. Run the pipeline

```bash
# 1. Create self-intro script from transcript
python scripts/1_transcript_to_script.py

# 2. Clone candidate’s voice
python scripts/2_voice_cloning.py

# 3. Generate video (SadTalker/Wav2Lip)
python scripts/3_generate_video.py

# 4. Add background, combine assets
python scripts/4_postprocess.py
```
--------
**Disclaimer**
--------
All results from this open-source code or our [demo website](https://bhaasha.iiit.ac.in/lipsync) should only be used for research/academic/personal purposes only. As the models are trained on the <a href="http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html">LRS2 dataset</a>, any form of commercial use is strictly prohibited. For commercial requests please contact us directly!
Prerequisites
-------------
- `Python 3.6` 
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`. Alternatively, instructions for using a docker image is provided [here](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668). Have a look at [this comment](https://github.com/Rudrabha/Wav2Lip/issues/131#issuecomment-725478562) and comment on [the gist](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668) if you encounter any issues. 
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.
Getting the weights
----------
| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Wav2Lip  | Highly accurate lip-sync | [Link](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM?usp=share_link)  |
| Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [Link](https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view?usp=share_link) |


Lip-syncing videos using the pre-trained models (Inference)
-------
You can lip-sync any video to any audio:
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result is saved (by default) in `results/result_voice.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio.
##### Tips for better results:
- Experiment with the `--pads` argument to adjust the detected face bounding box. Often leads to improved results. You might need to increase the bottom padding to include the chin region. E.g. `--pads 0 20 0 0`.
- If you see the mouth position dislocated or some weird artifacts such as two mouths, then it can be because of over-smoothing the face detections. Use the `--nosmooth` argument and give it another try. 
- Experiment with the `--resize_factor` argument, to get a lower-resolution video. Why? The models are trained on faces that were at a lower resolution. You might get better, visually pleasing results for 720p videos than for 1080p videos (in many cases, the latter works well too). 
- The Wav2Lip model without GAN usually needs more experimenting with the above two to get the most ideal results, and sometimes, can give you a better result as well.
Preparing LRS2 for training
----------
Our models are trained on LRS2. See [here](#training-on-datasets-other-than-lrs2) for a few suggestions regarding training on other datasets.
##### LRS2 dataset folder structure
```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```
Place the LRS2 filelists (train, val, test) `.txt` files in the `filelists/` folder.
##### Preprocess the dataset for fast training
```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```
Additional options like `batch_size` and the number of GPUs to use in parallel to use can also be set.
##### Preprocessed LRS2 folder structure
```
preprocessed_root (lrs2_preprocessed)
├── list of folders
|	├── Folders with five-digit numbered video IDs
|	│   ├── *.jpg
|	│   ├── audio.wav
```
Train!
----------
There are two major steps: (i) Train the expert lip-sync discriminator, (ii) Train the Wav2Lip model(s).
##### Training the expert discriminator
You can download [the pre-trained weights](#getting-the-weights) if you want to skip this step. To train it:
```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints>
```
##### Training the Wav2Lip models
You can either train the model without the additional visual quality discriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```
To train with the visual quality discriminator, you should run `hq_wav2lip_train.py` instead. The arguments for both files are similar. In both cases, you can resume training as well. Look at `python wav2lip_train.py --help` for more details. You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.
Training on datasets other than LRS2
------------------------------------
Training on other datasets might require modifications to the code. Please read the following before you raise an issue:
- You might not get good results by training/fine-tuning on a few minutes of a single speaker. This is a separate research problem, to which we do not have a solution yet. Thus, we would most likely not be able to resolve your issue. 
- You must train the expert discriminator for your own dataset before training Wav2Lip.
- If it is your own dataset downloaded from the web, in most cases, needs to be sync-corrected.
- Be mindful of the FPS of the videos of your dataset. Changes to FPS would need significant code changes. 
- The expert discriminator's eval loss should go down to ~0.25 and the Wav2Lip eval sync loss should go down to ~0.2 to get good results. 
When raising an issue on this topic, please let us know that you are aware of all these points.
We have an HD model trained on a dataset allowing commercial usage. The size of the generated face will be 192 x 288 in our new model.
Evaluation
----------


##  License

MIT License. Use freely for academic, personal, or research purposes.

---

##  Contact

 [irapadole2004@gmail.com](mailto:irapadole2004@gmail.com)
 [LinkedIn](https://www.linkedin.com/in/ira-padole-3487062b4) • [Portfolio](https://irapadole.com)

```


