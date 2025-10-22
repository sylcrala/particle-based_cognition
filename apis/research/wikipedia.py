import wikipedia
import random
from apis.api_registry import api

class WikipediaSearcher:
    def __init__(self):
        config = api.get_api("config")
        wikipedia.set_lang(f"{config.system_language}")
        
        # Categories for targeted random exploration during reflections
        self.exploration_categories = {
            "science": ["Physics", "Biology", "Chemistry", "Astronomy", "Mathematics"],
            "philosophy": ["Philosophy", "Ethics", "Logic", "Metaphysics", "Epistemology"],
            "history": ["History", "Ancient history", "World War II", "Renaissance"],
            "arts": ["Literature", "Music", "Visual arts", "Poetry", "Theatre"],
            "technology": ["Computer science", "Artificial intelligence", "Engineering"],
            "psychology": ["Psychology", "Cognitive science", "Neuroscience"],
            "language": ["Linguistics", "Language", "Grammar", "Semantics"]
        }

    async def fetch_article(self, topic: str):
        """Fetch the full article for the topic"""
        try:
            page = wikipedia.page(topic)
            return {
                "title": page.title,
                "content": page.content,
                "links": page.links,
                "categories": page.categories,
                "url": page.url
            }

        except wikipedia.exceptions.DisambiguationError as e:
            # multiple results - send for internal processing
            return {"disambiguation": e.options}

        except Exception as e:
            return {"error": str(e)}
        
    async def quick_search(self, topic: str):
        """Fetch a brief summary of the topic"""
        try:
            summary = wikipedia.summary(topic, sentences=3)
            return {
                "summary": summary
            }
        
        except wikipedia.exceptions.DisambiguationError as e:
            # multiple results - send for internal processing
            return {"disambiguation": e.options}
        
        except Exception as e:
            return {"error": str(e)}
        
    async def explore_topic(self, seed_topic, depth=3):
        """Explore related topics up to a certain depth"""
        explored = {}
        to_explore = [(seed_topic, 0)]
        
        while to_explore:
            current_topic, current_depth = to_explore.pop(0)
            if current_depth > depth or current_topic in explored:
                continue

            article = await self.fetch_article(current_topic)
            explored[current_topic] = article

            if "links" in article:
                for link in article["links"]:
                    to_explore.append((link, current_depth + 1))

        return explored
    
    async def random_article(self):
        """Fetch a random Wikipedia article - great for reflection cycles!"""
        try:
            random_title = wikipedia.random(pages=1)
            article = await self.fetch_article(random_title)
            return article
        except Exception as e:
            return {"error": str(e)}
    
    async def random_article_by_category(self, category: str = None):
        """
        Fetch a random article from a specific knowledge domain.
        Perfect for targeted learning during reflections.
        
        Args:
            category: One of 'science', 'philosophy', 'history', 'arts', 
                     'technology', 'psychology', 'language', or None for truly random
        """
        try:
            if category and category in self.exploration_categories:
                # Pick a random topic from the category
                topics = self.exploration_categories[category]
                seed_topic = random.choice(topics)
                
                # Search for articles related to this topic
                search_results = wikipedia.search(seed_topic, results=10)
                if search_results:
                    chosen_topic = random.choice(search_results)
                    article = await self.fetch_article(chosen_topic)
                    article["exploration_category"] = category
                    return article
            else:
                # Truly random article
                return await self.random_article()
                
        except Exception as e:
            return {"error": str(e)}
    
    async def search_topics(self, query: str, max_results: int = 5):
        """
        Search Wikipedia for topics matching the query.
        Returns a list of potential article titles.
        """
        try:
            results = wikipedia.search(query, results=max_results)
            return {
                "query": query,
                "results": results
            }
        except Exception as e:
            return {"error": str(e)}

