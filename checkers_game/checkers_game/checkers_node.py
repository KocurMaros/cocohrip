import re
import cv2
import pygame
from checkers_game.constants import WIDTH, HEIGHT, SQUARE_SIZE, RED, WHITE, BLACK
# use package-qualified import so the module is found when the package is installed
from checkers_game.camera.ximea_camera import XimeaCamera
from checkers_game.camera.board_detection import BoardDetection
from checkers_game.checkers.game import Game
from checkers_game.minimax.algorithm import WHITE, minimax
import mediapipe as mp
import time
import numpy as np

from checkers_msgs.msg import Board, Piece, Move, HandDetected, RobotMove

import rclpy
from rclpy.node import Node


class CheckersNode(Node):
    def __init__(self):
        super().__init__('checkers_game')
        print("\n" + "="*60)
        print("INITIALIZING CHECKERS NODE")
        print("="*60)
        
        pygame.init()
        self.WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Checkers")
        self.clock = pygame.time.Clock()
        self.FPS = 60
        
        print("Creating camera connection...")
        self.ximeaCamera = XimeaCamera()
        print("✓ Camera connected")
        
        print("\nInitializing board detection...")
        print("(This will open corner selection window)")
        self.boardDetection = BoardDetection(self.ximeaCamera)
        print("✓ Board detection ready")
        
        self.game = Game(self.WIN, self.boardDetection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.wait = False
        self.isRobotMoveDone = True

        self.board_publisher = self.create_publisher(Board, 'board_topic', 10)
        self.move_publisher = self.create_publisher(Move, 'move_topic', 10)
        self.hand_detected_publisher = self.create_publisher(HandDetected, 'hand_detected', 10)
        self.robot_move_subscription = self.create_subscription(RobotMove, '/robot_move_topic', self.robot_move_callback, 10)

        self.oldCameraImage = self.ximeaCamera.get_camera_image()
        self.waiting_for_true = True
        self.last_false_time = None
        self.isPlayerMoveDone = False
        self.game_started = False  # Add flag for manual game start

        print("\n" + "="*60)
        print("SYSTEM READY - Starting main game loop")
        print("="*60)
        print("\n→ Press 'S' in any OpenCV window to START the game")
        print("→ Press SPACE during gameplay to force move validation\n")
        
        # Create a ROS2 timer to handle pygame events every 1/FPS seconds
        self.timer = self.create_timer(1/self.FPS, self.update_pygame)


    def robot_move_callback(self, msg):
        print("\n✓ Robot finished its move")
        self.isRobotMoveDone = msg.robot_move_done

    def update_pygame(self):
        # This is your game's main loop controlled by ROS2 timer
        if pygame.event.get(pygame.QUIT):
            self.destroy_node()
            pygame.quit()
            return

        self.clock.tick(self.FPS)
        cameraImage = self.ximeaCamera.get_camera_image()
        board_coordinates = self.get_board_coordinates()
        isHandDetected = self.detect_hand(cameraImage, board_coordinates)
        boardDetected = self.boardDetection.get_board(cameraImage, self.game)

        if isHandDetected:
            self.publish_hand_detected_state()

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'S' to start the game
        if key == ord('s') or key == ord('S'):
            if not self.game_started:
                self.game_started = True
                print("\n" + "="*60)
                print("GAME STARTED - Make your first move!")
                print("="*60)
                print("\nYOU ARE WHITE (bottom 3 rows)")
                print("You can move WHITE pieces diagonally forward")
                print("\nValid starting positions for WHITE:")
                print("  Row 5: positions 40, 41, 42, 43")
                print("  Row 6: positions 48, 49, 50, 51")
                print("  Row 7: positions 56, 57, 58, 59")
                print("\nEach piece can move diagonally forward to an empty square")
                print("Example: Piece at position 40 → can move to position 32 or 33")
                print("\n→ Move a piece, then remove your hand completely")
                print("→ Wait for board to stabilize (1 second)")
                print("="*60 + "\n")
                self.waiting_for_true = True
                self.last_false_time = None
        
        # Press SPACE to force move validation
        if key == 32:
            print("\n[SPACE pressed] Forcing move validation...")
            self.wait = False
            self.isPlayerMoveDone = True  # Force validation
        
        # Only process game logic if game has started
        if not self.game_started:
            return
            
        isChange = self.findDifference(cameraImage, self.oldCameraImage)
        self.oldCameraImage = cameraImage

        if self.isRobotMoveDone:
            if self.waiting_for_true:
                if isChange:
                    self.waiting_for_true = False
                    print("\n→ Movement detected on board, waiting for it to stabilize...")
                    
                    # Show what's currently on the board in real-time
                    black_count = np.count_nonzero(boardDetected == 2)
                    white_count = np.count_nonzero(boardDetected == 1)
                    empty_count = np.count_nonzero(boardDetected == 0)
                    print(f"  Current detection: Black={black_count}, White={white_count}, Empty={empty_count}")
            else:
                if isChange:
                    # Reset the last_false_time if movement detected again
                    if self.last_false_time is not None:
                        print("  Movement resumed, restarting timer...")
                    self.last_false_time = None
                else:
                    if self.last_false_time is None:
                        self.last_false_time = time.time()
                        print("  Board stable, starting 1-second confirmation timer...")
                        
                        # Show current board state while waiting
                        black_count = np.count_nonzero(boardDetected == 2)
                        white_count = np.count_nonzero(boardDetected == 1)
                        empty_count = np.count_nonzero(boardDetected == 0)
                        print(f"  Current detection: Black={black_count}, White={white_count}, Empty={empty_count}")
                        
                    elif time.time() - self.last_false_time > 1:
                        print("✓ Board stable for 1 second, validating move...")
                        self.isPlayerMoveDone = True
                        self.last_false_time = None
                        self.waiting_for_true = True
                
            if self.isPlayerMoveDone or not self.wait:
                if self.game.winner() is not None:
                    print("\n" + "="*60)
                    print(f"GAME OVER - Winner: {self.game.winner()}")
                    print("="*60 + "\n")
                
                # Validate the move
                print("\n→ Validating move against game rules...")
                is_valid = self.game.wasItValidMove(boardDetected)
                
                if is_valid:
                    print("✓ Valid move detected!")
                    self.game.update(boardDetected)
                    self.publish_board_state()
                    
                    if self.game.board.isBoardCreated == True and self.game.turn == BLACK and self.game.aiMove == None:
                        print("\n" + "-"*60)
                        print("AI's turn - Calculating best move (depth 3)...")
                        value, new_board, new_move, new_piece, removed = minimax(self.game.get_board(), 3, BLACK, self)
                        print(f"✓ AI decided to move from ({new_piece.row},{new_piece.col}) to {new_move}")
                        self.aiMove = new_move
                        self.aiPiece = new_piece
                        self.aiPiece.color = BLACK
                        self.game.update(new_board.board)
                        self.publish_move_state(new_move, new_piece, removed)
                        self.isRobotMoveDone = False
                        print("Waiting for robot to execute move...")
                        print("-"*60 + "\n")
                    else:
                        print("\n→ Your move complete! Waiting for next move...\n")
                else:
                    # Count pieces properly for numpy array
                    black_count = np.count_nonzero(boardDetected == 2)
                    white_count = np.count_nonzero(boardDetected == 1)
                    empty_count = np.count_nonzero(boardDetected == 0)
                    
                    print("\n" + "="*60)
                    print("✗ INVALID MOVE - Move rejected")
                    print("="*60)
                    print(f"  Detected pieces: Black={black_count}, White={white_count}, Empty={empty_count}")
                    print(f"  Current turn: {'WHITE (you)' if self.game.turn == WHITE else 'BLACK (AI)'}")
                    
                    # Analyze what changed
                    self._analyze_board_changes(boardDetected)
                    
                    if self.game.turn == WHITE:
                        print("\n  YOUR TURN - Move a WHITE piece:")
                        try:
                            # Get all WHITE pieces
                            white_pieces = self.game.board.get_all_pieces_by_color(WHITE)
                            
                            if white_pieces and len(white_pieces) > 0:
                                # Calculate valid moves for each piece
                                movable_pieces = []
                                for piece in white_pieces:
                                    valid_moves = self.game.board.get_valid_moves(piece)
                                    if valid_moves and len(valid_moves) > 0:
                                        movable_pieces.append((piece, valid_moves))
                                
                                if len(movable_pieces) > 0:
                                    print(f"  You have {len(movable_pieces)} WHITE pieces that can move:")
                                    for piece, moves in movable_pieces:
                                        row, col = piece.row, piece.col
                                        position = row * 8 + col
                                        move_positions = [move[0] * 8 + move[1] for move in moves.keys()]
                                        print(f"    • Piece at position {position} (row {row}, col {col}) → can move to: {move_positions}")
                                else:
                                    print("  ⚠ No valid moves available for WHITE pieces!")
                            else:
                                print("  ⚠ No WHITE pieces found on board!")
                        except Exception as e:
                            print(f"  ⚠ Error getting valid moves: {e}")
                            print("  → Try moving any WHITE piece (cyan) diagonally forward")
                    
                    print("="*60 + "\n")
                    
                self.isPlayerMoveDone = False
                self.wait = True

    def findDifference(self, newCameraImage, oldCameraImage):
        image2 = oldCameraImage
        image1 = newCameraImage

        diff = cv2.absdiff(image1, image2)

        # Threshold the diff image so we get only significant changes
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Optionally apply some dilation to make the differences more visible
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Convert to grayscale if the images are color images
        if len(thresh.shape) == 3:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        # Calculate the percentage of change
        change_percentage = np.sum(thresh == 255) / thresh.size * 100

        # Increase threshold for significant change to reduce false positives
        significant_change_threshold = 2.0

        isChange = False
        if change_percentage > significant_change_threshold:
            message = f"Change: {change_percentage:.1f}%"
            isChange = True
        else:
            message = f"Stable: {change_percentage:.1f}%"
            isChange = False

        # Display the result with text
        cv2.putText(thresh, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Difference', thresh)

        return isChange

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def get_board_coordinates(self):
        board_coordinates = []
        for row in range(len(self.boardDetection.gameBoardFieldsContours)):
            for col in range(len(self.boardDetection.gameBoardFieldsContours[row])):
                boundary = self.boardDetection.gameBoardFieldsContours[row][col]
                # boundary is already [x, y, w, h], no need for cv2.boundingRect
                x, y, w, h = boundary
                centerX = x + w / 2
                centerY = y + h / 2
                board_coordinates.append([centerX, centerY])
        return board_coordinates    

    def detect_hand(self, cameraImage, board_coordinates):
        bgr_frame = cv2.cvtColor(cameraImage, cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(rgb_frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                min_x, min_y, max_x, max_y = self.get_hand_bounding_box(landmarks, rgb_frame)
                cv2.rectangle(rgb_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                cv2.imshow("Hand Tracking", rgb_frame)
                if self.do_rectangles_overlap((min_x, min_y, max_x - min_x, max_y - min_y), board_coordinates):
                    return True
                else:
                    return False
        cv2.imshow("Hand Tracking", rgb_frame)
        return False

    @staticmethod
    def do_rectangles_overlap(rect1, rect2, threshold=50):
        if len(rect1) != 4 or len(rect2) != 4:
            return False
        
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return (x1 - threshold < x2 + w2) and (x1 + w1 + threshold > x2) and (y1 - threshold < y2 + h2) and (y1 + h1 + threshold > y2)

    @staticmethod
    def get_hand_bounding_box(landmarks, frame):
        min_x = int(min(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
        max_x = int(max(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
        min_y = int(min(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
        max_y = int(max(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
        return min_x, min_y, max_x, max_y


    def color_to_string(self, color):
        if color == WHITE:
            return "white"
        elif color == BLACK:
            return "red"
        
    def publish_board_state(self):    
        board_msg = Board()
        for piece in self.game.board.get_all_pieces():
            piece_msg = Piece()
            piece_msg.row = piece.row
            piece_msg.col = piece.col
            piece_msg.color = self.color_to_string(piece.color)
            piece_msg.king = piece.king
            board_msg.pieces.append(piece_msg)
        self.board_publisher.publish(board_msg)

    def publish_move_state(self, new_move, new_piece, removed):
        move_msg = Move()
        move_msg.target_row = new_move[0]
        move_msg.target_col = new_move[1]

        piece_msg = Piece()
        piece_msg.row = new_piece.row
        piece_msg.col = new_piece.col
        piece_msg.color = self.color_to_string(new_piece.color)
        piece_msg.king = new_piece.king
        move_msg.piece_for_moving = piece_msg
        
        for piece in removed:
            piece_msg = Piece()
            piece_msg.row = piece.row
            piece_msg.col = piece.col
            piece_msg.color = self.color_to_string(piece.color)
            piece_msg.king = piece.king
            move_msg.removed_pieces.append(piece_msg)

        self.move_publisher.publish(move_msg)

    def publish_hand_detected_state(self):
        hand_detected_msg = HandDetected()
        hand_detected_msg.hand_detected = True
        # self.hand_detected_publisher.publish(hand_detected_msg)

    def _analyze_board_changes(self, new_board_detected):
        """
        Analyze what changed between the old board state and new detected state
        to provide helpful feedback about the attempted move
        """
        if not self.game.board.board or not self.game.board.isBoardCreated:
            print("\n  → Board not initialized yet")
            return
        
        old_board = self.game.board.board
        
        # PRINT CURRENT BOARD STATE FOR DEBUGGING
        self._print_board_state(old_board, "CURRENT BOARD IN MEMORY")
        self._print_board_detection(new_board_detected, "DETECTED BOARD FROM CAMERA")
        
        # Find pieces that disappeared (potential source positions)
        disappeared_pieces = []
        for row in range(8):
            for col in range(8):
                old_piece = old_board[row][col]
                new_piece = new_board_detected[row][col]
                
                # Piece was there, now it's gone
                if old_piece != 0 and new_piece == 0:
                    position = row * 8 + col
                    disappeared_pieces.append((row, col, position, old_piece))
        
        # Find pieces that appeared (potential destination positions)
        appeared_pieces = []
        for row in range(8):
            for col in range(8):
                old_piece = old_board[row][col]
                new_piece = new_board_detected[row][col]
                
                # Empty before, piece now
                if old_piece == 0 and new_piece != 0:
                    position = row * 8 + col
                    color = "WHITE" if new_piece == 1 else "BLACK"
                    appeared_pieces.append((row, col, position, color))
        
        print("\n  MOVE ANALYSIS:")
        
        if len(disappeared_pieces) == 0 and len(appeared_pieces) == 0:
            print("    ⚠ No changes detected on board")
            print("    → Board comparison:")
            
            # Show both boards for debugging
            old_white = sum(1 for row in old_board for piece in row if piece != 0 and hasattr(piece, 'color') and piece.color == WHITE)
            old_black = sum(1 for row in old_board for piece in row if piece != 0 and hasattr(piece, 'color') and piece.color == BLACK)
            new_white = np.count_nonzero(new_board_detected == 1)
            new_black = np.count_nonzero(new_board_detected == 2)
            
            print(f"      Old board: WHITE={old_white}, BLACK={old_black}")
            print(f"      New board: WHITE={new_white}, BLACK={new_black}")
            print("    → This might mean board states are identical or detection failed")
            return
        
        # Show what disappeared
        if len(disappeared_pieces) > 0:
            print(f"    Pieces removed from board:")
            for row, col, pos, piece in disappeared_pieces:
                color = "WHITE" if piece.color == WHITE else "BLACK"
                
                # Get valid moves for this piece BEFORE it was moved
                valid_moves = self.game.board.get_valid_moves(piece)
                move_positions = [move[0] * 8 + move[1] for move in valid_moves.keys()]
                
                print(f"      • {color} piece at position {pos} (row {row}, col {col})")
                if len(move_positions) > 0:
                    print(f"        Valid moves were: {move_positions}")
                else:
                    print(f"        ⚠ This piece had NO valid moves!")
        
        # Show what appeared
        if len(appeared_pieces) > 0:
            print(f"    Pieces added to board:")
            for row, col, pos, color in appeared_pieces:
                print(f"      • {color} piece at position {pos} (row {row}, col {col})")
        
        # Try to match source → destination
        if len(disappeared_pieces) == 1 and len(appeared_pieces) == 1:
            src_row, src_col, src_pos, src_piece = disappeared_pieces[0]
            dst_row, dst_col, dst_pos, dst_color = appeared_pieces[0]
            
            # Check if colors match
            src_color = "WHITE" if src_piece.color == WHITE else "BLACK"
            
            if src_color == dst_color:
                print(f"\n    ATTEMPTED MOVE:")
                print(f"      {src_color} from position {src_pos} → position {dst_pos}")
                print(f"      From (row {src_row}, col {src_col}) → (row {dst_row}, col {dst_col})")
                
                # Check if this destination was valid
                valid_moves = self.game.board.get_valid_moves(src_piece)
                valid_destinations = list(valid_moves.keys())
                
                if (dst_row, dst_col) in valid_destinations:
                    print(f"      ✓ Destination WAS in valid moves!")
                    print(f"      → But the move was still rejected - check game logic")
                else:
                    print(f"      ✗ Destination NOT in valid moves!")
                    move_positions = [move[0] * 8 + move[1] for move in valid_moves.keys()]
                    print(f"      → Valid moves were: {move_positions}")
            else:
                print(f"\n    ⚠ Color mismatch: {src_color} piece removed, {dst_color} piece added")
        
        elif len(disappeared_pieces) > 1:
            print(f"\n    ⚠ Multiple pieces removed ({len(disappeared_pieces)}) - possible capture or hand blocking view")
        
        elif len(appeared_pieces) > 1:
            print(f"\n    ⚠ Multiple pieces added ({len(appeared_pieces)}) - possible detection error")

    def _print_board_state(self, board, title):
        """Print the board state from game memory (Piece objects)"""
        print(f"\n  {title}:")
        print("  " + "="*65)
        print("      0   1   2   3   4   5   6   7")
        print("    " + "-"*65)
        
        for row in range(8):
            row_str = f"  {row} |"
            for col in range(8):
                piece = board[row][col]
                position = row * 8 + col
                
                if piece == 0:
                    row_str += f" {position:2d} |"
                elif hasattr(piece, 'color'):
                    if piece.color == WHITE:
                        row_str += f" W{position:2d}|"
                    else:
                        row_str += f" B{position:2d}|"
                else:
                    row_str += f" ?{position:2d}|"
            print(row_str)
        
        print("    " + "-"*65)
        print("  Legend: W=WHITE (you), B=BLACK (AI), ##=empty position")
        
        # Print piece positions summary
        white_positions = []
        black_positions = []
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != 0 and hasattr(piece, 'color'):
                    position = row * 8 + col
                    if piece.color == WHITE:
                        white_positions.append(position)
                    else:
                        black_positions.append(position)
        
        print(f"\n  WHITE pieces at: {white_positions}")
        print(f"  BLACK pieces at: {black_positions}")
        print("  " + "="*65)

    def _print_board_detection(self, board_detected, title):
        """Print the detected board state (numeric values: 0=empty, 1=white, 2=black)"""
        print(f"\n  {title}:")
        print("  " + "="*65)
        print("      0   1   2   3   4   5   6   7")
        print("    " + "-"*65)
        
        for row in range(8):
            row_str = f"  {row} |"
            for col in range(8):
                value = board_detected[row][col]
                position = row * 8 + col
                
                if value == 0:
                    row_str += f" {position:2d} |"
                elif value == 1:
                    row_str += f" W{position:2d}|"
                elif value == 2:
                    row_str += f" B{position:2d}|"
                else:
                    row_str += f" ?{position:2d}|"
            print(row_str)
        
        print("    " + "-"*65)
        print("  Legend: W=WHITE (you), B=BLACK (AI), ##=empty position")
        
        # Print piece positions summary
        white_positions = []
        black_positions = []
        for row in range(8):
            for col in range(8):
                value = board_detected[row][col]
                position = row * 8 + col
                if value == 1:
                    white_positions.append(position)
                elif value == 2:
                    black_positions.append(position)
        
        print(f"\n  WHITE pieces at: {white_positions}")
        print(f"  BLACK pieces at: {black_positions}")
        print("  " + "="*65)

def main(args=None):
    rclpy.init(args=args)
    checkers_node = CheckersNode()
    rclpy.spin(checkers_node)

    checkers_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()