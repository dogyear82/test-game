#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "MannyPlatformerGameMode.generated.h"

UCLASS()
class MANNYPLATFORMER_API AMannyPlatformerGameMode : public AGameModeBase
{
    GENERATED_BODY()

public:
    AMannyPlatformerGameMode();

protected:
    virtual void BeginPlay() override;
};

