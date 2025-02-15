# **Segment Anything GUI - Tkinter Interface**  
This project provides a **Tkinter-based GUI** for using **Meta’s Segment Anything Model (SAM)**.  
The application allows users to **import an image, specify segmentation inputs** (points, bounding boxes, or text prompts),  
and visualize the **segmented output**. A **loading bar** is included to indicate when the model is processing.  
<img width="912" alt="Screenshot 2025-02-15 at 13 30 54" src="https://github.com/user-attachments/assets/aa9a9019-01d5-4b9b-b1de-a7b0bb9f97fe" />
<img width="912" alt="Screenshot 2025-02-15 at 13 31 06" src="https://github.com/user-attachments/assets/fff9a320-7147-48bf-9a10-839d41a2db88" />

---

## **📥 Download the Model**
Before running the GUI, download the **Segment Anything Model (SAM)** checkpoint:  

- **ViT-H (default)** → [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)  
- **ViT-L** → [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)  
- **ViT-B** → [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)  

Move the downloaded `.pth` file to the project directory.

---

## **🎯 Usage**
### **1️⃣ Run the GUI**
Run the script:
```bash
python sam_gui.py
