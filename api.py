# from fastapi import FastAPI, UploadFile, File
# from typing import List
# import cv2
# import numpy as np

# from infer import recognize_odometer_two_stage

# app = FastAPI(title="Odometer Recognition API (Two Stage)")

# @app.post("/api/odometer/recognize-batch")
# async def recognize_odometer_batch(
#     images: List[UploadFile] = File(...)
# ):
#     raw_results = []

#     for idx, image in enumerate(images):
#         contents = await image.read()
#         img = cv2.imdecode(
#             np.frombuffer(contents, np.uint8),
#             cv2.IMREAD_COLOR
#         )

#         if img is None:
#             continue

#         result = recognize_odometer_two_stage(img)
#         result["filename"] = image.filename
#         result["index"] = idx
#         raw_results.append(result)

#     return {
#         "count": len(images),
#         "results": raw_results
#     }

from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np

from infer import recognize_odometer_two_stage, merge_two_stage_results

app = FastAPI(title="Odometer Recognition API (Two Stage)")

@app.post("/api/odometer/recognize-batch")
async def recognize_odometer_batch(
    images: List[UploadFile] = File(...)
):
    raw_results = []

    for idx, image in enumerate(images):
        contents = await image.read()
        img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            continue

        result = recognize_odometer_two_stage(img)
        result["filename"] = image.filename
        result["index"] = idx
        raw_results.append(result)

    valid_results = [r for r in raw_results if r.get("success")]
    final_result = merge_two_stage_results(valid_results)

    return {
        "count": len(images),
        "final_result": final_result,
        "results": raw_results
    }


