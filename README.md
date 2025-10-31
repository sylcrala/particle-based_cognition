# The Particle-based Cognition Engine
*A new perspective on artificial intelligence*


## Summary

This project is an attempt to build a new type of neural network designed entirely around experiential learning - building knowledge through experiences. 


## Installation Instructions

To install and set up this repository, please run the following:
```
    cd \dir\to\install\to\
    git clone https://github.com/sylcrala/particle-based_cogntion.git
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
```

Prior to your first launch, please navigate to ./shared_services/config.py and update any required settings

If you plan on using the engine in "cog-growth" mode, where all knowledge accumulation happens through experiential growth, then you may skip the following steps.

Otherwise, if you plan on using the engine in "llm-extension" mode, please download your preferred model and place the directory within ./models/, then rename the models directory to "core"
 - The file path for your model directory should look like this: "<directory you installed to>/models/core/<all your model files here>"


## Launch Instructions

There are two methods for launching

 - The main GUI, with a 3D field visualizer and other features:
    ```python main.py```

 - Or the direct console launcher, which bypasses the GUI launch:
    ```python main_direct.py```


## Background

This all started around the fall of 2024, when I downloaded and set up my first local model, Mistral 7b Instruct v02 (the same model mainly used for testing), from Huggingface and began down my path of local AI development. 

I originally wanted to create my own memory system for locally installed LLMs, to establish a persistent and consistently growing memory and identity across sessions; but when I started to work on this, I learned that LLMs are designed to have their memory/weights be updated offline (fine-tuning). This led me to the idea "What if there was an architecture that allowed for dynamic real-time weight updates?" and here I am a year later attempting to build one.

I must warn you, I have no formal background in this department. I didn't go to college for computer science, as much as I wish I did. I've been working with VSC Copilot (Claude Sonnet 4) throughout the creation of this project, and you will see a combination of AI generated and my own created code, but this has mainly been an educational journey for me.

This project was made over the course of a year and the combination and revision of multiple other projects, all created under the same goals and intention (we love adhd).


## Disclaimer

This project is in active development, and it is *nowhere near complete*. You are likely to find bugs and issues, and the GUI is currently not finished either. You're probably not going to encounter coherent speech from the agent, unless you're in llm-extension mode. Cog-growth mode currently is in very early linguistic development, but if you leave a session going long enough and watch the console + parse the logs, you'll see the development of language over time; Each session normally starts out with very probabilistic character generation based on particle/neuron location, frequency, valence, etc; but over time as more words and phrases are encountered this heavily changes; even when those phrases aren't fed back into the available alphabet + word pool for generation. 

I'll be adding more diagnostic and analytical tools for observing the underlying behavior at play


## Contribution

I'd absolutely *love* your feedback and contributions! Please don't heistate to open a pull request or submit an issue, and if you want to collaborate please reach out!


## Licensing

This project is licensed under:
- **[AGPLv3](LICENSE.md)** - Base license terms
- **[Additional Terms](TERMS.md)** - Section 7 restrictions and requirements
- **[Ethics Policy](ETHICS.md)** - Values and intent

**Quick Summary**: Open source for research/personal use. Commercial use requires separate licensing after careful consideration. No military/surveillance applications.


--


For questions or concerns, please reach out to <contact@sylcrala.xyz>
