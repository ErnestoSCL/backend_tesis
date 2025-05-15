from pydantic import BaseModel
from typing import List, Union

class InputArray(BaseModel):
    values: List[Union[int, float, str]]


