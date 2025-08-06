from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import requests
from io import BytesIO
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI()

# Load RealESRGAN Model
model_path = 'models/RealESRGAN_x4.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=torch.device(device)
)

@app.get("/enhance/")
async def enhance_image(image_url: str = Query(...)):
    try:
        # Download image
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Convert to numpy array and enhance
        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=4)

        # Convert to JPEG stream
        buffer = BytesIO()
        Image.fromarray(output).save(buffer, format="JPEG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
