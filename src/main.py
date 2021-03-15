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

# Images
# dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
# imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batched list of images

# # Inference
# results = model(imgs)
# results.print()  # or .show(), .save()
@app.post("/predict")
async def serving(file:UploadFile=File(...)):
    image = file.file
    image = io.imread(image)
    # print(type(image))
    results = model(image)
    r = {}
    colors = color_list()
    r['predict']={}
    j = 0
    for i, (img, pred) in enumerate(zip(results.imgs, results.pred)):
        # str = f'image {i + 1}/{len(results.pred)}: {img.shape[0]}x{img.shape[1]} '
        r["size"]= {"height":str(img.shape[0]),"width":str(img.shape[1])}
        if pred is not None:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                
                # str += f"{n} {results.names[int(c)]}{'s' * (n > 1)}, "
                r['predict'][j]= results.names[int(c)]
                j+=1
        for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{results.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
        f= Path("data").mkdir(exist_ok=True)
        f= Path("data") / "".join([results.names[int(c)],".jpeg"])
        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img 

        # f = Path("data") / results.files[i] 
        if f.is_file():
            j = 1 
            while(Path(f).is_file()):
                f = Path("data") / "".join([results.names[int(c)],str(j),".jpeg"])
                j+=1     
        img.save(f)
        r["image"]= "127.0.0.1:5000/"+str(f)
    return {"result":r}
@app.get("/data/{name}")
async def main(name):
    file_path = "data/{}".format(name)
    print(file_path)
    return FileResponse(file_path)
