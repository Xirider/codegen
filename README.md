# codegen

Codegen allows you to easily finetune GPT-2 Large (1.5 billion parameter) on code or other text. You can also then run inference on the model with just one command.

It uses gradient checkpointing and the AdaFactor Optimizer by default, which allows you to train GPT-2 Large on a single GPU with 16GB VRAM (like V100 or P100) without running out of memory.

## Usage

Run the notebook train.ipynp to install the necessary libraries, download the model and data and prepare the data. It also includes the steps to finetune the model and generate code afterwards.

Alternatively, you can run train.py after the data is prepared to perform the training without having to keep the notebook active.

## Results

This is an example function created after running the 1.5 billion parameter model for 14k iterations. The input conditioning was "def join_text(text_list):":

```python
def join_text(text_list):
        """"""Return a list of tuples (int3, int4) where (int3, int4) are the
        lengths of tuples formed by joining text (int3, int4)
        into a common string

        Args:
            text_list: a list of text tuples
        Returns:
            a list of tuples (int3, int4) where (int3, int4) are the
           lengths of tuples formed by joining text (int3, int4)
        """"""
        if isinstance(text_list, list):
            return text_list
        elif isinstance(text_list, (int, str)):
            return [Text(t) for t in text_list]
        else:
            return [Text(t) for t in text_list] 
```



