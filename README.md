Quick cog implementation of some ESRGAN upscalers pulled from https://upscale.wiki/wiki/Model_Database

To build locally, run:

    cog run scripts/pull_models.sh
    cog build
    cog predict -i image=@'path/to/your/image'
