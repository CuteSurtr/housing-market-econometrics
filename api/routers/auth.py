"""
Authentication router for the Housing Market Econometrics API.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import logging

from ..core.dependencies import get_db
from ..core.security import (
    security_manager, get_current_active_user, create_user_token,
    validate_password_strength, validate_email, sanitize_input
)
from ..models.crud import UserCRUD, APIKeyCRUD
from ..models.schemas import (
    User, UserCreate, UserUpdate, Token, LoginRequest, APIKey, APIKeyCreate,
    BaseResponse, ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=User)
def register_user(
    user: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user.
    """
    try:
        # Validate input
        user.username = sanitize_input(user.username)
        user.email = sanitize_input(user.email)
        
        # Validate email format
        if not validate_email(user.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Validate password strength
        password_validation = validate_password_strength(user.password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
            )
        
        # Check if user already exists
        existing_user = UserCRUD.get_by_username(db, user.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        existing_email = UserCRUD.get_by_email(db, user.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password and create user
        hashed_password = security_manager.get_password_hash(user.password)
        db_user = UserCRUD.create(db, user, hashed_password)
        
        logger.info(f"New user registered: {user.username}")
        return db_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login to get access token.
    """
    try:
        # Get user by username
        user = UserCRUD.get_by_username(db, form_data.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not security_manager.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        # Create access token
        access_token = create_user_token(user.username)
        
        logger.info(f"User logged in: {user.username}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 1800  # 30 minutes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/login/json", response_model=Token)
def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Login using JSON payload.
    """
    try:
        # Get user by username
        user = UserCRUD.get_by_username(db, login_data.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not security_manager.verify_password(login_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        # Create access token
        access_token = create_user_token(user.username)
        
        logger.info(f"User logged in via JSON: {user.username}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 1800  # 30 minutes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during JSON login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=User)
def get_current_user_info(
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current user information.
    """
    try:
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )


@router.put("/me", response_model=User)
def update_current_user(
    user_update: UserUpdate,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update current user information.
    """
    try:
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate email if provided
        if user_update.email:
            user_update.email = sanitize_input(user_update.email)
            if not validate_email(user_update.email):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid email format"
                )
            
            # Check if email is already taken by another user
            existing_email = UserCRUD.get_by_email(db, user_update.email)
            if existing_email and existing_email.id != user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already taken"
                )
        
        # Update user
        updated_user = UserCRUD.update(db, user.id, user_update)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
        
        logger.info(f"User updated: {current_user}")
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


# API Key management
@router.post("/api-keys", response_model=APIKey)
def create_api_key(
    api_key: APIKeyCreate,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new API key for the current user.
    """
    try:
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate API key
        api_key_value = security_manager.generate_api_key()
        key_hash = security_manager.get_password_hash(api_key_value)
        
        # Create API key record
        db_api_key = APIKeyCRUD.create(db, user.id, api_key, key_hash)
        
        # Return the actual API key (only shown once)
        response_data = {
            "id": db_api_key.id,
            "name": db_api_key.name,
            "user_id": db_api_key.user_id,
            "is_active": db_api_key.is_active,
            "created_at": db_api_key.created_at,
            "last_used": db_api_key.last_used,
            "api_key": api_key_value  # Only shown on creation
        }
        
        logger.info(f"API key created for user: {current_user}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/api-keys", response_model=List[APIKey])
def get_api_keys(
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all API keys for the current user.
    """
    try:
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        api_keys = APIKeyCRUD.get_by_user(db, user.id)
        return api_keys
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API keys"
        )


@router.delete("/api-keys/{key_id}", response_model=BaseResponse)
def delete_api_key(
    key_id: int,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete an API key.
    """
    try:
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if API key belongs to user
        api_key = APIKeyCRUD.get_by_id(db, key_id)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        if api_key.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this API key"
            )
        
        # Delete API key
        success = APIKeyCRUD.delete(db, key_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete API key"
            )
        
        logger.info(f"API key deleted for user: {current_user}")
        return {
            "success": True,
            "message": "API key deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key"
        )


# Password change
@router.post("/change-password", response_model=BaseResponse)
def change_password(
    current_password: str,
    new_password: str,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.
    """
    try:
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not security_manager.verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        password_validation = validate_password_strength(new_password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"New password validation failed: {', '.join(password_validation['errors'])}"
            )
        
        # Hash new password and update user
        new_hashed_password = security_manager.get_password_hash(new_password)
        user.hashed_password = new_hashed_password
        db.commit()
        
        logger.info(f"Password changed for user: {current_user}")
        return {
            "success": True,
            "message": "Password changed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


# Health check for authentication
@router.get("/health")
def auth_health_check():
    """
    Authentication service health check.
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": "2024-01-01T00:00:00Z"
    }
