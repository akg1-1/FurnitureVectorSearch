import logging
from typing import List, Dict, Any, Optional
import open_clip
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models


class FurnitureVectorSearch:
    """
    Векторный поиск мебели с использованием CLIP и Qdrant.
    
    Позволяет выполнять текстовый поиск по коллекции мебельных товаров
    с поддержкой фильтрации по типу, стилям и цветам.
    """

    def __init__(self,
                    model_name: str = "ViT-H-14",
                    pretrained: str = "dfn5b",
                    db_url: str = "http://localhost:6333",
                    logger: Optional[logging.Logger] = None
                ):
        """
        Инициализация поискового движка.

        Args:
            model_name: Название модели open_clip.
            pretrained: Веса предобучения.
            db_url: Адрес Qdrant (HTTP или gRPC).
            collection_name: Имя коллекции по умолчанию.
            logger: Логгер (если не передан, создаётся заглушка).
        """
        self.logger = logger or self._get_null_logger()    
        
        self.client = self.__init_connection(db_url)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.__init_embedding_model(model_name, pretrained)


    @staticmethod
    def _get_null_logger() -> logging.Logger:
        logger = logging.getLogger(__name__ + ".null")
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
        return logger


    def __init_connection(self,
                            db_url: str) -> QdrantClient:
        try:
            client = QdrantClient(url=db_url)
            client.get_collections()
            self.logger.info("Connected to Qdrant at %s", db_url)
            return client
        except Exception as e:
            self.logger.error("Qdrant connection failed: %s", e)
            return client


    def __init_embedding_model(self,
                                model_name: str,
                                pretrained: str):
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                device=self.device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            self.logger.info("Model %s loaded on %s", model_name, self.device)
            return model, tokenizer
        except Exception as e:
            self.logger.error("Model loading failed: %s", e)
            return None


    def check_health(self) -> bool:
        """Проверяет доступность Qdrant и модели."""
        try:    
            self.client.get_collections()
            self.logger.info("Health check OK")
            return True
        except Exception as e:
            self.logger.error("Health check failed: %s", e)
            return False


    def __build_filter(self, filter_dict: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Преобразует словарь фильтров в объект Qdrant Filter.
        Все текстовые значения полей type, styles, colors
        автоматически приводятся к нижнему регистру.
        """
        if not filter_dict:
            return None
        
        # Приводим значения фильтров к нижнему регистру,
        # чтобы они совпадали с payload (где мы уже всё привели к lower).
        if "type" in filter_dict and isinstance(filter_dict["type"], str):
            filter_dict["type"] = filter_dict["type"].lower()

        for key in ("styles", "colors"):
            if key in filter_dict and isinstance(filter_dict[key], list):
                filter_dict[key] = [
                    v.lower() if isinstance(v, str) else v
                    for v in filter_dict[key]
                ]
        try:
            conditions = []
            # 1. Фильтр по типу (точное совпадение)
            if "type" in filter_dict and filter_dict["type"]:
                conditions.append(
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value=filter_dict["type"])
                    )
                )
            # 2. Фильтр по стилям (любой из списка)
            if "styles" in filter_dict and filter_dict["styles"]:
                conditions.append(
                    models.FieldCondition(
                        key="styles",
                        match=models.MatchAny(any=filter_dict["styles"])
                    )
                )
            # 3. Фильтр по цветам (любой из списка)
            if "colors" in filter_dict and filter_dict["colors"]:
                conditions.append(
                    models.FieldCondition(
                        key="colors",
                        match=models.MatchAny(any=filter_dict["colors"])
                    )
                )

            self.logger.info("Build Qdrant filter success")
            return models.Filter(must=conditions) if conditions else None

        except Exception as e:
            self.logger.error("Build filter error %s", e)
            return None


    def search(self,
            query: str,
            collection_name: str,
            limit: int = 5,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List:
        """
        Выполняет текстовый поиск по коллекции.

        Args:
            query: Текстовый запрос (например, "диван синий скандинавский").
            collection_name: Имя коллекции (если не указано, используется из __init__).
            limit: Максимальное количество результатов.
            filters: Словарь фильтров (см. _build_filter).

        Returns:
            Список словарей с ключами: id, score, payload.
        """
        try:
            with torch.no_grad():
                text_tokens = self.tokenizer([query]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_vector = text_features[0].cpu().tolist()
        
            qdrant_filter = self.__build_filter(filters) if filters else None
        
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,     
                with_vectors=False
            )
            
            results = []
            for point in search_result.points:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                })
            return results
        except Exception as e:
            self.logger.error(e)
            return None


    def search_with_fallback(self,
                    query: str,
                    collection_name: str,
                    limit: int = 5,
                    filters: Optional[Dict[str, Any]] = None,
                    ) -> Optional[List[Dict[str, Any]]]:
        """
        Выполняет поиск с резервным запросом без фильтров, если первый поиск не дал результатов.

        Сначала пробует поиск с переданными фильтрами. Если ничего не найдено,
        выполняет повторный поиск по тому же запросу, но без фильтров.

        Args:
            query: Текстовый запрос (например, "диван синий скандинавский").
            collection_name: Имя коллекции.
            limit: Максимальное количество результатов.
            filters: Словарь фильтров (см. _build_filter). Если None, фильтры не применяются.

        Returns:
            Список словарей с ключами: id, score, payload.
            Если произошла ошибка, возвращает None.
        """
        try:
            first_search = self.search(
                query=query,
                collection_name=collection_name,
                limit=limit,
                filters=filters
            )
            if first_search:
                return first_search
            else:
                # Повторяем поиск без фильтров
                second_search = self.search(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    filters=None   # явно без фильтров
                )
                return second_search
        except Exception as e:
            self.logger.error(f"Ошибка в multiply_search: {e}")
            return None