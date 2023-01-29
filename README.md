# Style Transfer telegram bot

## What is it? 
This is a telegram bot with style transfer for input image (content image) and style image🌁. 

The principle of operation📃: 

You should send to bot two images. 
1. Content image - image, which you want to transform🌆. 
2. Style image - image in which style you want to get🎨.  

✅After that bot send you transformed your content image with style, like in image.

### 📖Style transfer bot contains two different networks:
<details>
 <summary>Slow model 🐢</summary>

The model based on pretrained vgg19 network, where we try to train the input image in order to minimise the content/style losses ([more info](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)).

**Advantage:** 

High quality of style transfer process. This type of model works with any style and give you very interesting result.

**Disadvantage:** 

Long training process, so you must several minutes for get the result.
</details> 


<details>
 <summary>Fast model 🚀</summary>

This type of model based on [this paper](https://arxiv.org/abs/1705.06830). We use pretrained network on a corpus of roughly 80,000 paintings, so we don't need to train model ([more info](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)).

**Advantage:** 

High speed of work. We get the result image after few seconds. 

**Disadvantage:** 

Inasmuch as this model pretrained on certain paintings, you can't get high quality result for all your style images. 
</details>  

## How to install it and start? 

1. Clone the repository on your local directory   
2. Set the bot **token** in file **config.py**
3. Start the file **telegram_bot.py**

## How it works?

**1️⃣  Open the bot**

<img src="mockups\1.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

**2️⃣  Send the **/start** message** 

<img src="mockups\2.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

**3️⃣  Choose the type of transformation (slow or fast):** 

<img src="mockups\2_1.png" alt="MarineGEO circle logo" style="height: 80px; width:320px;"/>

<details>
<summary>Slow transformation</summary>

**4️⃣  Send the content image** 

<img src="mockups\3.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

**5️⃣  Send the style image** 

<img src="mockups\4.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

**6️⃣  Get the result, after few minutes** 

<img src="mockups\5.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

</details>  

<details>
<summary>Fast transformation</summary>

**4️⃣  Send the content image** 

<img src="mockups\6.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

**5️⃣  Send the style image** 

<img src="mockups\7.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

**6️⃣  Get the result after several seconds** 

<img src="mockups\8.png" alt="MarineGEO circle logo" style="height: 736px; width:365px;"/>

</details>  


✔️ **Ready!** 




