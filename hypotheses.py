BASELINE_FEATURES = [
    "order_price",
    "city",
    "platform_id",
    "delivery_service",
    "is_pod",
    "category_name",
    "microcat_name",
    "payment_count",
    "payment_total",
    "payment_method_count",
    "payment_method_first",
]


HYPOTHESES = {
    "baseline": {
        "description": "Базовая модель на простых признаках заказа, товара, доставки и оплаты.",
        "conclusion": "Базовая точка отсчёта для сравнения всех остальных гипотез.",
        "features": BASELINE_FEATURES,
    },

    "seller_platform_age": {
        "description": "Проверяем, влияет ли стаж продавца на платформе на вероятность возврата.",
        "conclusion": "Есть предварительный вывод: с увеличением срока пребывания продавца на платформе количество возвратов уменьшается.",
        "features": BASELINE_FEATURES + [
            "seller_registrationtime",
            "seller_firstlistingdate",
            "seller_account_age_days",
            "seller_firstlisting_age_days",
        ],
    },

    "buyer_platform_age": {
        "description": "Проверяем, влияет ли стаж покупателя на платформе на вероятность возврата.",
        "conclusion": "Есть предварительный вывод: у покупателей зависимость выглядит более случайной, явного устойчивого паттерна пока не видно.",
        "features": BASELINE_FEATURES + [
            "buyer_registrationtime",
            "buyer_firstlistingdate",
            "buyer_account_age_days",
            "buyer_firstlisting_age_days",
        ],
    },

    "seller_history": {
        "description": "Проверяем, влияет ли история продавца на вероятность возврата: сколько заказов было у продавца и сколько из них закончились возвратом.",
        "conclusion": "Есть предварительный вывод: с ростом количества предыдущих заказов количество возвратов уменьшается.",
        "features": BASELINE_FEATURES + [
            "seller_orders_count",
            "seller_returns_count",
            "seller_return_rate",
            "seller_return_rate_7d",
            "seller_return_rate_14d",
            "seller_return_rate_30d",
        ],
    },

    "buyer_history": {
        "description": "Проверяем, влияет ли история покупателя на вероятность возврата: сколько заказов было у покупателя и сколько из них закончились возвратом.",
        "conclusion": "Есть предварительный вывод: с ростом количества предыдущих заказов количество возвратов уменьшается.",
        "features": BASELINE_FEATURES + [
            "buyer_orders_count",
            "buyer_returns_count",
            "buyer_return_rate",
            "buyer_return_rate_7d",
            "buyer_return_rate_14d",
            "buyer_return_rate_30d",
        ],
    },

    "user_block_status": {
        "description": "Проверяем, влияет ли статус блокировки покупателя или продавца на вероятность возврата.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "buyer_isblocked",
            "seller_isblocked",
        ],
    },

    "buyer_microcat_experience": {
        "description": "Проверяем, влияет ли опыт покупателя в конкретной микрокатегории на вероятность возврата. Идея: если покупатель впервые берёт товар новой микрокатегории, риск возврата может быть выше.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "buyer_microcat_orders_count",
            "buyer_microcat_returns_count",
            "buyer_microcat_return_rate",
            "buyer_is_first_microcat_order",
        ],
    },

    "delivery_service_quality": {
        "description": "Проверяем, влияет ли служба доставки на вероятность возврата.",
        "conclusion": "Есть предварительный вывод: по службам доставки видны лучшие и худшие сервисы по доле возвратов.",
        "features": BASELINE_FEATURES + [
            "delivery_service",
            "delivery_service_orders_count",
            "delivery_service_returns_count",
            "delivery_service_return_rate",
        ],
    },

    "price_effect": {
        "description": "Проверяем, влияет ли цена товара/заказа на вероятность возврата.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "order_price",
            "price_bucket",
        ],
    },

    "items_count_in_order": {
        "description": "Проверяем, влияет ли количество товаров в одном заказе на вероятность возврата.",
        "conclusion": "Есть предварительный вывод: если товаров в заказе больше, это не обязательно означает больше возвратов.",
        "features": BASELINE_FEATURES + [
            "items_count_in_order",
            "same_category_items_count_in_order",
            "same_microcat_items_count_in_order",
        ],
    },

    "company_vs_private": {
        "description": "Проверяем, влияет ли тип пользователя: компания или частное лицо.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "buyer_iscompany",
            "seller_iscompany",
        ],
    },

    "rare_category": {
        "description": "Проверяем, влияет ли редкость категории или микрокатегории на вероятность возврата.",
        "conclusion": "Есть предварительный вывод: гипотеза подтверждается, чем более редкая категория, тем меньше возвратов.",
        "features": BASELINE_FEATURES + [
            "category_orders_count",
            "microcat_orders_count",
            "category_frequency",
            "microcat_frequency",
            "is_rare_category",
            "is_rare_microcat",
        ],
    },

    "delivery_waiting_time": {
        "description": "Проверяем, влияет ли долгое ожидание или задержка между созданием и принятием заказа на вероятность возврата.",
        "conclusion": "Гипотеза сейчас непроверяема в честном виде: найден leakage через order_accept_date. Если использовать order_accept_date, модель получает почти идеальный score, потому что этот признак связан с фактом невозврата. Поэтому order_accept_date не используем в модели.",
        "features": BASELINE_FEATURES,
    },

    "platform_effect": {
        "description": "Проверяем, влияет ли платформа оформления заказа на вероятность возврата: iOS, Android, web, mobile web.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "platform_id",
        ],
    },

    "buyer_seller_location": {
        "description": "Проверяем, влияет ли локация покупателя и продавца на вероятность возврата. В текущем датасете явно есть city по локации покупателя, но отдельной локации продавца может не хватать.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "city",
        ],
    },

    "seller_vs_buyer_bad_actors": {
        "description": "Проверяем, кто сильнее связан с возвратами: продавцы с плохой историей или покупатели с плохой историей.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "seller_orders_count",
            "seller_returns_count",
            "seller_return_rate",
            "buyer_orders_count",
            "buyer_returns_count",
            "buyer_return_rate",
        ],
    },

    "item_price_share_in_order": {
        "description": "Проверяем, влияет ли доля цены конкретного товара относительно общей суммы заказа на вероятность возврата.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "item_price_share_in_order",
            "items_count_in_order",
        ],
    },

    "bad_actors_blacklist": {
        "description": "Проверяем идею поиска недобросовестных продавцов и покупателей, чтобы отделять их влияние от обычной статистики пользователей.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "seller_is_bad_actor",
            "buyer_is_bad_actor",
            "seller_return_rate",
            "buyer_return_rate",
        ],
    },

    "regional_effect": {
        "description": "Проверяем региональные особенности: в каких городах или регионах возвратов больше или меньше.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "city",
            "city_orders_count",
            "city_returns_count",
            "city_return_rate",
        ],
    },

    "title_text_features": {
        "description": "Проверяем, влияет ли текст названия товара на вероятность возврата.",
        "conclusion": "Выводов не было.",
        "features": BASELINE_FEATURES + [
            "title",
            "title_len",
        ],
    },
}


ACTIVE_HYPOTHESES = [
    "baseline",
    "seller_platform_age",
    "buyer_platform_age",
    "seller_history",
    "buyer_history",
    "user_block_status",
    "buyer_microcat_experience",
    "delivery_service_quality",
    "price_effect",
    "items_count_in_order",
    "company_vs_private",
    "rare_category",
    "delivery_waiting_time",
    "platform_effect",
    "buyer_seller_location",
    "seller_vs_buyer_bad_actors",
    "item_price_share_in_order",
    "bad_actors_blacklist",
    "regional_effect",
    "title_text_features",
]


LEAKAGE_FORBIDDEN_FEATURES = [
    "order_accept_date",
    "cancel_date",
]