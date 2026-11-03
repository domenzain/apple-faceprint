# Demo image provenance

The repository owner supplied `examples/assets/reference.png`, an AI-generated portrait, and explicitly approved it for public use. The three other portraits were created with the built-in image-generation tool using the prompts below. Image metadata was removed before publication. The face crops and pipeline figure are outputs of the code in this repository.

## Matching variation

Input: `examples/assets/reference.png` as a subject reference.

> Use case: identity-preserve. Asset: a single square 1024×1024 synthetic portrait for a public face-embedding code example. The attached AI-generated portrait is the authorized subject reference. Make a second photograph of the SAME adult man, preserving his facial proportions, curly hair, full beard and thin round glasses. Change pose to nearly frontal with a slight turn toward camera right, soft expression, no pipe or hand near the face; dark jacket over a plain shirt. Keep a monochrome 1930s studio-photograph aesthetic, realistic skin texture and soft side lighting, plain dark-gray background, head and shoulders with the entire head visible and modest margin. This should be a distinct composition, not a pixel copy. One person, no text, no borders, no watermark.

Output: [`examples/assets/match.png`](../examples/assets/match.png).

## Fictional person B

> Use case: photorealistic-natural. Asset: one square 1024×1024 monochrome portrait for a public face-embedding software demonstration. Generate an entirely fictional adult man about forty: clean-shaven, straight short hair swept to one side with a high hairline, a broad square jaw, small deep-set eyes, a long straight nose, no glasses. Three-quarter head-and-shoulders studio portrait, facing slightly toward camera left, entire head visible with modest margin. Dark textured wool jacket and plain shirt, plain charcoal-gray backdrop, soft directional studio light, realistic skin and subtle film grain, 1930s photographic aesthetic. One person only. No resemblance to a named or famous person is intended. No text, borders or watermark.

Output: [`examples/assets/person-b.png`](../examples/assets/person-b.png).

## Fictional person C

> Use case: photorealistic-natural. Asset: one square 1024×1024 monochrome portrait for a public face-embedding software demonstration. Generate an entirely fictional adult woman about forty: straight dark chin-length bob with blunt fringe, gently rounded face, wide-set eyes, defined cheekbones, small slightly upturned nose, no glasses. Nearly frontal head-and-shoulders studio portrait with a subtle turn toward camera right, entire head visible with modest margin. Simple dark wool jacket over a high-neck blouse, plain charcoal-gray background, soft directional studio lighting, realistic skin texture and subtle film grain, 1930s photographic aesthetic. One person only. No resemblance to a named or famous person is intended. No text, borders or watermark.

Output: [`examples/assets/person-c.png`](../examples/assets/person-c.png).
