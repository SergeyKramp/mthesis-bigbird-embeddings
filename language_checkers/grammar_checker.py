import language_tool_python
from typing import List


class GrammarChecker:
    """A wrapper class for the grammar checker. Used for getting grammar mistake features.
    """

    def __init__(self) -> None:
        self.tool = None
        self.initialized = False 
    
    def init(self) -> None:
        """initialize the grammar checker.
        """
        self.tool = language_tool_python.LanguageTool('en-US')
        self.initialized = True
        
    def get_grammar_mistake_rules(self, text: str) -> List[str]:
        """get a list of grammar mistake rules for a given text.

        Args:
            text (str): a text to be checked.

        Returns:
            list: a list of grammar mistake rules.
        """
        if not self.initialized:
            self.tool = language_tool_python.LanguageTool('en-US')
            self.initialized = True

        matches = [match.ruleId for match in self.tool.check(text)]

        return matches
  
    @property
    def url(self) -> str:
        """get the url of the grammar checker.

        Returns:
            str: the url of the grammar checker.
        """
        return self.tool._url
    
    def close(self) -> None:
        """close the grammar checker.
        """
        if not self.initialized:
            return
        
        self.tool.close()
        self.initialized = False


