from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    @abstractmethod
    def select(self, users, slot_info):
        """
        Bir slot için hangi kullanıcıya ne kadar kaynak ayrılacağını döndürür.
        Dönüş: list of (user_id, rb_count)
        users: list[User]
        slot_info: dict (örneğin toplam RB sayısı, kanal kazanımı)
        """
        raise NotImplementedError
