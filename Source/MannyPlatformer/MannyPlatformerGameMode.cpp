#include "MannyPlatformerGameMode.h"
#include "MannyCharacter.h"
#include "GoalTrigger.h"

#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/PlayerStart.h"
#include "UObject/ConstructorHelpers.h"

AMannyPlatformerGameMode::AMannyPlatformerGameMode()
{
    DefaultPawnClass = AMannyCharacter::StaticClass();
}

void AMannyPlatformerGameMode::BeginPlay()
{
    Super::BeginPlay();

    UWorld* World = GetWorld();
    if (!World)
    {
        return;
    }

    // Load the cube mesh from Engine BasicShapes
    static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeMeshFinder(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
    UStaticMesh* CubeMesh = CubeMeshFinder.Succeeded() ? CubeMeshFinder.Object : nullptr;

    auto SpawnPlatform = [&](const FVector& Location, const FVector& Scale)
    {
        if (!CubeMesh) return (AStaticMeshActor*)nullptr;
        FActorSpawnParameters Params;
        Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AStaticMeshActor* Platform = World->SpawnActor<AStaticMeshActor>(Location, FRotator::ZeroRotator, Params);
        if (Platform)
        {
            Platform->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
            Platform->SetActorScale3D(Scale);
            Platform->GetStaticMeshComponent()->SetMobility(EComponentMobility::Static);
            Platform->GetStaticMeshComponent()->SetCollisionProfileName(TEXT("BlockAll"));
        }
        return Platform;
    };

    // Ground / first platform
    AStaticMeshActor* P1 = SpawnPlatform(FVector(0.f, 0.f, 25.f), FVector(4.f, 4.f, 0.5f));

    // Second platform (forward and higher)
    AStaticMeshActor* P2 = SpawnPlatform(FVector(600.f, 0.f, 125.f), FVector(4.f, 4.f, 0.5f));

    // Third (goal) platform
    AStaticMeshActor* P3 = SpawnPlatform(FVector(1200.f, 0.f, 225.f), FVector(4.f, 4.f, 0.5f));

    // Goal trigger above third platform
    if (P3)
    {
        const FVector P3Loc = P3->GetActorLocation();
        FActorSpawnParameters Params;
        Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AGoalTrigger* Goal = World->SpawnActor<AGoalTrigger>(P3Loc + FVector(0.f, 0.f, 80.f), FRotator::ZeroRotator, Params);
        if (Goal)
        {
            // Scale trigger to cover the platform area
            Goal->SetActorScale3D(FVector(4.f, 4.f, 2.f));
        }
    }

    // Ensure player spawns in a good spot
    if (APlayerController* PC = UGameplayStatics::GetPlayerController(World, 0))
    {
        if (APawn* Existing = PC->GetPawn())
        {
            Existing->Destroy();
        }
        const FTransform StartXform(FRotator::ZeroRotator, FVector(-200.f, 0.f, 300.f));
        RestartPlayerAtTransform(PC, StartXform);
    }
}

