"""Dummy VOTS tracker — returns initialization mask for every frame."""
from trax import Image, Region, Server, TraxStatus

def main():
    objects = None
    with Server([Region.MASK, Region.POLYGON, Region.RECTANGLE], [Image.PATH], multiobject=True) as server:
        while True:
            request = server.wait()
            if request.type in [TraxStatus.QUIT, TraxStatus.ERROR]:
                break
            if request.type == TraxStatus.INITIALIZE:
                objects = request.objects
            server.status(objects)

if __name__ == "__main__":
    main()
