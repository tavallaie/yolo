import torch
from fastapi import FastAPI, File, UploadFile, Query
from skimage import io, transform
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.responses import FileResponse
from ploty import plot_one_box , color_list

app = FastAPI(
    title="Yolov5 RestApi",
    description="Automatic API for Yolo5 large dataset",
    version="0.0.1",
)
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)


@app.post("/predict")
async def serving(file:UploadFile=File(...)):
    try:    
        image = file.file
        image = io.imread(image)
        results = model(image)
        r = {}
        colors = color_list()
        r['result']={}
        j = 0

        for i, (img, pred) in enumerate(zip(results.imgs, results.pred)):

            r["size"]= {"height":str(img.shape[0]),"width":str(img.shape[1])}
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    
            for *box, conf, cls in pred:  # xyxy, confidence, class
                            label = f'{results.names[int(cls)]} {conf:.2f}'
                            plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
                            r['result'][j] ={"predictions":results.names[int(cls)],"conf":round(float(conf),3)}
                            j+=1
            f= Path("data").mkdir(exist_ok=True)
            f= Path("data") / "".join([results.names[int(c)],".jpeg"])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img 

            if f.is_file():
                j = 1 
                while(Path(f).is_file()):
                    f = Path("data") / "".join([results.names[int(c)],str(j),".jpeg"])
                    j+=1     
            img.save(f)
            r["image"]= "http://127.0.0.1:5000/"+str(f)
        return r
    except Exception as e:
        return {"error":"some Error happend. please try with another image"}
@app.get("/data/{name}")
async def main(name):
    file_path = "data/{}".format(name)
    print(file_path)
    return FileResponse(file_path)
