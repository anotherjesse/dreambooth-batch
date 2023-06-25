import base64
import requests
import sys
import json


def gen(output_fn, **kwargs):
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        for idx, datauri in enumerate(data["output"]):
            base64_encoded_data = datauri.split(",")[1]
            data = base64.b64decode(base64_encoded_data)
            with open(f"{idx}-{output_fn}", "wb") as f:
                f.write(data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)


def main():
    gen(
        "sample.batch.png",
        images=json.dumps(
            [
                {
                    "name": "txt2img",
                    "inputs": {
                        "prompt": "a macro photograph of male bfirsh black arts movement magic realism funk art by Gino Severini, Ric Estradamirror shades, ray - tracing, sexy gaze, one light",
                        "negative_prompt": "childish, poorly drawn, ugly",
                        "num_outputs": 4,
                        "scheduler": "PNDM",
                        "disable_safety_check": False,
                        "seed": 13510,
                    },
                },
                {
                    "name": "img2img",
                    "inputs": {
                        "prompt": "a macro photograph of male bfirsh black arts movement magic realism funk art by Gino Severini, Ric Estradamirror shades, ray - tracing, sexy gaze, one light",
                        "negative_prompt": "childish, poorly drawn, ugly",
                        "image": "https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
                        "scheduler": "KerrasDPM",
                        "disable_safety_check": False,
                        "seed": 42,
                    },
                },
            ]
        ),
        weights="https://replicate.delivery/pbxt/BxsckHvjQWpyGZL7vj2nn7N8lLi7ATfGVY7YtIErz4utTCBIA/output.zip",
    )


if __name__ == "__main__":
    main()
