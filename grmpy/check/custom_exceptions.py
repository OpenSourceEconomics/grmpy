"""This module contains the custom exceptions for the package."""


class UserError(Exception):
    """ This custom error class provides informative feedback in case of a misspecified
    request by the user.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "\n\n         {}\n\n".format(self.msg)
