import openai
import json
import logging

################################### USER AGENT POUR NAVIGATION ###################################


# module logging ########################################################################

NAME_LOG = "journal_assistant_openai.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s"
)

file_handler = logging.FileHandler(NAME_LOG)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#################################### HISTORIQUE DES REQUETES #####################################################

# "write a meta description for a website whose themes are : xx, xx, xx"

# "find reformulations for the following sentence : 'xxxxxxx' "

######################### CONSTANTES ###############################

## OPEN AI

openai.organization = "xxxxxxxxxxxxxx"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxx"

NB_TOKEN_REQUETE = 1500  # integer, max 1500


####### requete

requete = "write a meta description for a webpage whose theme is : xx xx"

######################### FONCTIONS OPENAI #########################


def assistant_openai(requete_cible=None):

    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{requete_cible}",
            temperature=0.8,
            max_tokens=NB_TOKEN_REQUETE,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        cast_json = json.loads(str(response))

        reponse_openai = str(cast_json["choices"][0]["text"]).strip()

        logger.debug(f"réponse de l'assistant : {reponse_openai}")

        logger.debug("--------------------------------------------------------")

        return True

    except Exception as e:
        logger.exception(e)


def assistant_graphique(requete_cible=None):

    try:
        response = openai.Image.create(
            prompt=f"{requete_cible}", n=1, top_p=1.0, size="1024x1024"
        )

        image_url = response["data"][0]["url"]

        logger.debug(f"réponse de l'assistant : {image_url}")

        logger.debug("--------------------------------------------------------")

        return True

    except Exception as e:
        logger.exception(e)


####################################################################


####################### MAIN #######################

if __name__ == "__main__":

    try:
        assistant_openai(requete_cible=requete)

    except Exception as e:
        logger.exception(e)

#####################################################
